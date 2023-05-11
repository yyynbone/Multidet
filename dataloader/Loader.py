from multiprocessing.pool import Pool
import torch
import random
import numpy as np
import cv2
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader

from dataloader.load import select_image, select_video, load_images, load_labels, load_image_result, \
    rect_shape, load_mosaic, read_image

from dataloader.transform import Transfrom, gray
from dataloader.augmentations import mask_label
from utils import print_log, xyxy2xywh


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

def get_dataset(path, pre_process, img_size=640, batch_size=16, logger=None, image_weights=False,
            single_cls=False, stride=32, pad=0.0, prefix='', bgr=True, filter_str='', select_class=(),
             filter_bkg=False):
    if isinstance(img_size, int):
        img_size = (img_size, img_size)   # h, w
    elif len(img_size)==1:
        img_size = (img_size[0], img_size[0])
    # else:
    #     # img_size is (w,h) ,now we convert to (h,w)
    #     img_size = tuple((img_size[1], img_size[0]))
    rect = False if image_weights else pre_process.get('rect', False)
    augment = pre_process.get('augment', False)

    pre_process['rect'] = rect
    pre_process['augment'] = augment

    # select sample randomly
    portion = pre_process.get('sample_portion', 1)

    img_files = select_image(path, filter_str=filter_str, remove_str='_mask', sample=portion, bg_gain=1, obj_gain=1)
    assert img_files, f'{prefix}No images found'
    print_log(f'now we found {len(img_files)} images in dataset', logger)

    results = load_images(img_files, img_size=img_size)

    # cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')
    # try:
    #     np.save(path, results)  # save cache for next time
    #     path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
    #     print_log(f'{prefix}New cache created: {path}', logger)
    # except Exception as e:
    #     print_log(f'{prefix}WARNING: Cache directory {path.parent} is not writeable: {e}', logger)  # not writeable

    if pre_process.get('label', False):
        results = load_labels(results, select_class=select_class, prefix=prefix, filter_bkg=filter_bkg, logger=logger)
        if single_cls:
            for result in results:
                result['labels'][:, 0] = 0
                if result['instance_segments'] is not None:
                    result['instance_segments'][:, 0] = 0
        if rect:
            results = rect_shape(results, pad, stride, batch_size)
    pre_process['img_size'] = img_size
    pre_process['mosaic'] = pre_process.get('mosaic', 0.3) if augment and not rect else 0 # load 4 images at a time into a mosaic (only during training)
    pre_process["gray"] = not bgr
    return results, pre_process


class LoadImagesAndLabels(Dataset):
    #  loads images and labels for training and validation
    """
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
        self.labels = [result['labels'] for result in results]

    def __init__(self, path, pre_process, img_size=640, batch_size=16, logger=None, image_weights=False,
        single_cls=False, stride=32, pad=0.0, prefix='',bgr=True, filter_str='', select_class=(),
         filter_bkg=False, bkg_ratio=0, **kwargs):
        if isinstance(img_size, int):
            img_size = (img_size, img_size)   # h, w
        elif len(img_size)==1:
            img_size = (img_size[0], img_size[0])
        # else:
        #     # img_size is (w,h) ,now we convert to (h,w)
        #     img_size = tuple((img_size[1], img_size[0]))
        self.img_size = img_size
        self.gray = not bgr
        rect = False if image_weights else pre_process.get('rect', False)
        self.mask_labels = pre_process.get('mask_label', False)
        augment = pre_process.get('augment', False)
        self.mosaic = augment and not rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_p = pre_process.get('mosaic', 0)
        pre_process['rect'] = rect
        pre_process['augment'] = augment

        # select sample randomly
        portion = pre_process.get('sample_portion', 1)

        self.img_files = select_image(path, filter_str=filter_str, sample=portion, bg_gain=1, obj_gain=1)
        assert self.img_files, f'{prefix}No images found'
        print_log(f'now we found {len(self.img_files)} images in dataset', logger)

        results = load_images(self.img_files, img_size=img_size)

        # cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')
        # try:
        #     np.save(path, results)  # save cache for next time
        #     path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
        #     print_log(f'{prefix}New cache created: {path}', logger)
        # except Exception as e:
        #     print_log(f'{prefix}WARNING: Cache directory {path.parent} is not writeable: {e}', logger)  # not writeable

        if pre_process.get('label', True):
            results = load_labels(results, select_class=select_class, prefix=prefix, filter_bkg=filter_bkg, logger=logger)
            if single_cls:
                for result in results:
                    result['labels'][:, 0] = 0
                    if result['instance_segments'] is not None:
                        result['instance_segments'][:, 0] = 0
            if rect:
                results = rect_shape(results, pad, stride, batch_size)
            self.labels = [result['labels'] for result in results]

        self.T = Transfrom(pre_process, logger)
        # for result in results:
        #     self.T(result)
        #     labels = result['labels']
        #     if labels is not None:
        #         result['wh'] = np.vstack([labels[:, 3] - labels[:, 1], labels[:, 4] - labels[:, 2]]).T

        self.results = results
        self.indices = list(range(len(self.results)))
        random.shuffle(self.indices)
        self.bkg_ratio = bkg_ratio
    """
    def __init__(self, pre_process, results, logger=None, bkg_ratio=0, obj_mask=0):

        self.mask_labels = pre_process.get('mask_label', False)
        self.mosaic = pre_process.get('mosaic', 0)
        self.gray = pre_process.get('gray')

        if pre_process.get('label', False):
            self.labels = [result['labels'] for result in results]
            label_num = np.array([len(l) for l in self.labels])
            self.obj_index = np.where(label_num > 0)[0]
            self.bkg_index = np.where(label_num == 0)[0]

        self.T = Transfrom(pre_process, logger)
        # for result in results:
        #     self.T(result)
        #     labels = result['labels']
        #     if labels is not None:
        #         result['wh'] = np.vstack([labels[:, 3] - labels[:, 1], labels[:, 4] - labels[:, 2]]).T

        self.results = results
        self.indices = list(range(len(self.results)))

        self.bkg_ratio = bkg_ratio
        self.obj_mask = obj_mask


    def format(self, result):
        trans_result = deepcopy(result)
        self.T(trans_result)
        if self.gray:
            gray(trans_result)
        assert not isinstance(trans_result['labels'], list), f'error, format, {trans_result}'
        labels = trans_result['labels']
        class_label = None
        if labels is not None:
            # result['wh'] = np.vstack([labels[:, 3]-labels[:, 1], labels[:, 4]-labels[:, 2]]).T
            class_label = np.zeros((1, 1))
            # # labels to cxcywh
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5], w=trans_result['img'].shape[1], h=trans_result['img'].shape[0], clip=True,
                                       eps=1E-3)
            if labels.shape[0]:
                class_label += 1

        # Convert
        img = trans_result['img']
        img = img.transpose((2, 0, 1))  # HWC to CHW
        img = np.ascontiguousarray(img[::-1])  # BGR to RGB
        return img, labels, class_label, trans_result

    def index_shuffle(self):
        self.indices = np.arange(len(self.results))
        if self.bkg_ratio:
            indices = self.obj_index
            obj_num = len(indices)
            bkg_indices = np.random.choice(self.bkg_index, int(obj_num*self.bkg_ratio))
            self.indices = np.concatenate((indices, bkg_indices))
        if self.obj_mask:
            mask_indices = -(1 + self.obj_index)
            self.indices = np.concatenate((self.indices, mask_indices))
        # self.indices = -(1 + self.obj_index)
        random.shuffle(self.indices)

    def __len__(self):
        return len(self.indices)


    def __getitem__(self, index):
        # return self.results[self.indices[index]]
        pic_index = self.indices[index]
        if pic_index < 0:
            pic_index = -pic_index-1
            result = deepcopy(self.results[pic_index])
            read_image(result)
            mask_label(result)
        else:
            if random.random() < self.mosaic:
                result = load_mosaic(self.results, pic_index)
            else:
                result = deepcopy(self.results[pic_index])
                read_image(result)

        return self.format(result)

    @staticmethod
    def collate_fn(batch):
        imgs = []
        segs = []
        class_labels = []
        im_labels = []
        paths = []
        # shapes = []
        for i, (img, labels, class_label, result) in enumerate(batch):
            img = torch.from_numpy(img.copy())
            if class_label is not None:
                class_label = torch.from_numpy(class_label)

            if labels is not None:
                im_label = torch.zeros((len(labels),6))
                im_label[:, 0] = i
                im_label[:, 1:] = torch.from_numpy(labels)
            else:
                im_label = labels
            path = result['filename']
            if isinstance(path, list):
                path = path[0][:-4]+'_mosaic.jpg'
            #     shape = (None,None)
            # else:
            #     shape = (result['ori_shape'], (result['resized_shape'], (0, 0)))
            seg = result['segment']
            if seg is not None:
                seg = torch.from_numpy(seg)
            segs.append(seg)
            imgs.append(img)
            class_labels.append(class_label)
            im_labels.append(im_label)
            paths.append(path)
            # shapes.append(shape)

        class_labels = None  if None in class_labels else torch.cat(class_labels, 0)
        im_labels = None if None in im_labels else torch.cat(im_labels, 0)
        segs = None if None in segs else torch.stack(segs, 0)

        col_target = [class_labels, im_labels, segs]
        return torch.stack(imgs, 0), col_target, paths# , shapes

class LoadImages(LoadImagesAndLabels):
    def __init__(self, path, img_size=640, stride=32, auto=True, is_bgr=True):
        pre_process = {'transform': {"resize": None,'adapt_pad':{'auto':auto, 'rectangle_stride': stride}} }
        self.gray = not is_bgr
        if isinstance(img_size, int):
            img_size = (img_size, img_size)  # h, w
        elif len(img_size) == 1:
            img_size = (img_size[0], img_size[0])
        images = select_image(path)
        videos = select_video(path)

        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.auto = auto
        self.is_bgr = is_bgr
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {path}.'
        # super().__init__(pre_process, results)
        self.T = Transfrom(pre_process)


    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            while not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            s = f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: '

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, f'Image Not Found {path}'
            s = f'image {self.count}/{self.nf} {path}: '

        result = load_image_result(img0, self.img_size)
        img, labels, class_label, trans_result = self.format(result)

        return path, img, img0, self.cap, s

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files

class LoadStreams:
    # streamloader, i.e. `python detect.py --source 'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP streams`
    def __init__(self, logger=None, sources='streams.txt', img_size=640, stride=32, auto=True):
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride
        self.logger =logger

    #     if os.path.isfile(sources):
    #         with open(sources) as f:
    #             sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
    #     else:
    #         sources = [sources]
    #
    #     n = len(sources)
    #     self.imgs, self.fps, self.frames, self.threads = [None] * n, [0] * n, [0] * n, [None] * n
    #     self.sources = [clean_str(x) for x in sources]  # clean source names for later
    #     self.auto = auto
    #     for i, s in enumerate(sources):  # index, source
    #         # Start thread to read frames from video stream
    #         st = f'{i + 1}/{n}: {s}... '
    #         # if 'youtube.com/' in s or 'youtu.be/' in s:  # if source is YouTube video
    #         #     check_requirements(('pafy', 'youtube_dl'))
    #         #     import pafy
    #         #     s = pafy.new(s).getbest(preftype="mp4").url  # YouTube URL
    #         s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
    #         cap = cv2.VideoCapture(s)
    #         assert cap.isOpened(), f'{st}Failed to open {s}'
    #         w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #         h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #         self.fps[i] = max(cap.get(cv2.CAP_PROP_FPS) % 100, 0) or 30.0  # 30 FPS fallback
    #         self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')  # infinite stream fallback
    #
    #         _, self.imgs[i] = cap.read()  # guarantee first frame
    #         self.threads[i] = Thread(target=self.update, args=([i, cap, s]), daemon=True)
    #         self.logger.info(f"{st} Success ({self.frames[i]} frames {w}x{h} at {self.fps[i]:.2f} FPS)")
    #         self.threads[i].start()
    #     self.logger.info('')  # newline
    #
    #     # check for common shapes
    #     s = np.stack([letterbox(x, self.img_size, stride=self.stride, auto=self.auto)[0].shape for x in self.imgs])
    #     self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
    #     if not self.rect:
    #         self.logger.warning('WARNING: Stream shapes differ. For optimal performance supply similarly-shaped streams.')
    #
    # def update(self, i, cap, stream):
    #     # Read stream `i` frames in daemon thread
    #     n, f, read = 0, self.frames[i], 1  # frame number, frame array, inference every 'read' frame
    #     while cap.isOpened() and n < f:
    #         n += 1
    #         # _, self.imgs[index] = cap.read()
    #         cap.grab()
    #         if n % read == 0:
    #             success, im = cap.retrieve()
    #             if success:
    #                 self.imgs[i] = im
    #             else:
    #                 LOGGER.warning('WARNING: Video stream unresponsive, please check your IP camera connection.')
    #                 self.imgs[i] = np.zeros_like(self.imgs[i])
    #                 cap.open(stream)  # re-open stream if signal was lost
    #         time.sleep(1 / self.fps[i])  # wait time
    #
    # def __iter__(self):
    #     self.count = -1
    #     return self
    #
    # def __next__(self):
    #     self.count += 1
    #     if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord('q'):  # q to quit
    #         cv2.destroyAllWindows()
    #         raise StopIteration
    #
    #     # Letterbox
    #     img0 = self.imgs.copy()
    #     img = [letterbox(x, self.img_size, stride=self.stride, auto=self.rect and self.auto)[0] for x in img0]
    #
    #     # Stack
    #     img = np.stack(img, 0)
    #
    #     # Convert
    #     img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
    #     img = np.ascontiguousarray(img)
    #
    #     return self.sources, img, img0, None, ''
    #
    # def __len__(self):
    #     return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years
