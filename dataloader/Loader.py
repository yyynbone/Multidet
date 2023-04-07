from multiprocessing.pool import Pool
import torch
import random
from torch.utils.data import Dataset, DataLoader

from dataloader.load import select_image, load_images, load_labels, rect_shape, transforms
from utils import print_log


class InfiniteDataLoader(DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

class _RepeatSampler:
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

# class Loader:
#     def __init__(self):

class LoadImagesAndLabels(Dataset):
    #  loads images and labels for training and validation
    def __init__(self, path,  img_size=640, batch_size=16, logger=None, augment=False, hyp=None, rect=False, image_weights=False,
                single_cls=False, stride=32, pad=0.0, prefix='',is_bgr=True, filter_str='', select_class=(),
                 filter_bkg=False):
        if isinstance(img_size, int):
            img_size = (img_size, img_size)   # h, w
        elif len(img_size)==1:
            img_size = (img_size[0], img_size[0])
        else:
            # img_size is (w,h) ,now we convert to (h,w)
            img_size = tuple((img_size[1], img_size[0]))
        if is_bgr:
            hyp['to_gray'] = 0
        else:
            hyp['to_gray'] = 1
        self.hyp = hyp
        rect = False if image_weights else rect
        self.mosaic = augment and not rect  # load 4 images at a time into a mosaic (only during training)

        # select sample randomly
        portion = 1
        if hyp is not None:
            if 'train' in prefix:
                portion = hyp.get('train_sample_portion', 1)
            elif 'val' in prefix:
                portion = hyp.get('val_sample_portion', 1)
        self.img_files = select_image(path, filter_str=filter_str, sample=portion, bg_gain=1, obj_gain=1)
        assert self.img_files, f'{prefix}No images found'
        print_log(f'now we found {len(self.img_files)} images in dataset', logger)

        results = load_images(self.img_files, img_size=img_size)
        results = load_labels(results, select_class=select_class, prefix=prefix, filter_bkg=filter_bkg, logger=logger)
        if single_cls:
            for result in results:
                result['labels'][:, 0] = 0
                if result['instance_segments'] is not None:
                    result['instance_segments'][:, 0] = 0
        if rect:
            results = rect_shape(results, pad, stride, batch_size)

        results =  transforms(results, hyp=hyp, augment=augment, logger=logger)

        self.results = results

        self.indices = list(range(len(self.results)))
        random.shuffle(self.indices)

    def __len__(self):
        return len(self.img_files)


    def __getitem__(self, index):
        return self.results[self.indices[index]]

    @staticmethod
    def collate_fn(batch):
        imgs = []
        segs = []
        class_labels = []
        im_labels = []
        paths = []
        shapes = []
        for i, result in enumerate(batch):
            img = torch.from_numpy(result['img'])
            class_label = result['class_label']
            if class_label is not None:
                class_label = torch.from_numpy(result['class_label'])

            labels = result['labels']
            if labels is not None:
                im_label = torch.zeros((len(labels),6))
                im_label[:, 0] = i
                im_label[:, 1:] = torch.from_numpy(labels)
            else:
                im_label = labels
            path = result['filename']
            if isinstance(path, list):
                path = path[0][:-4]+'_mosaic.jpg'
                shape = (None,None)
            else:
                shape = (result['ori_shape'], (result['resized_shape'], (0, 0)))
            seg = result['segment']
            if seg is not None:
                seg = torch.from_numpy(seg)
            segs.append(seg)
            imgs.append(img)
            class_labels.append(class_label)
            im_labels.append(im_label)
            paths.append(path)
            shapes.append(shape)

        class_labels = None  if None in class_labels else torch.cat(class_labels, 0)
        im_labels = None if None in im_labels else torch.cat(im_labels, 0)
        segs = None if None in segs else torch.stack(segs, 0)

        col_target = [class_labels,im_labels, segs]
        return torch.stack(imgs, 0), col_target, paths, shapes
