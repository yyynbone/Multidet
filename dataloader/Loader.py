import glob
import os
import random
from pathlib import Path
import cv2
import numpy as np
import time
from threading import Thread
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from dataloader.check_load import IMG_FORMATS, VID_FORMATS, load_mosaic, load_image

from dataloader.augmentations import Albumentations, augment_hsv, mixup, random_perspective, adapt_pad, resize, xy2wh, flip, gray, format

from utils import (set_logging, xywh2xyxy, xyxy2xywh, clean_str,segments2boxes, print_log)
from copy import deepcopy


LOGGER = set_logging(__name__)

def glob_file(path):
    p = str(Path(path).resolve())  # os-agnostic absolute path
    if '*' in p:
        p_glob = list(glob.glob(p, recursive=True))  # glob
    elif os.path.isdir(p):
        p_glob = list(glob.glob(p+'/**/*.*', recursive=True))
    elif os.path.isfile(p):
        if '.txt' in p:
            with open(p) as t:
                t = t.read().strip().splitlines()
                parent = str(Path(p).parent) + os.sep   # os.sep能够在不同系统上采用不同的分隔符
                p_glob = [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
        else:
            p_glob = [p]  # files
    else:
        raise Exception(f'ERROR: {p} does not exist')
    return p_glob

def select_format(f, file_format=IMG_FORMATS):
    f = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in file_format)
    return f

def select_sample(paths, filter_str='', sample=1, bg_gain=1, obj_gain=1):
    # select image data which contains filter_str
    if filter_str:
        paths = [im_f for im_f in paths if filter_str in im_f]
    files = []
    bg_sample = min(sample * bg_gain, 1)
    obj_sample = min(sample * obj_gain, 1)
    for f in paths:
        rand_num = random.randint(0,100)/100
        if str(Path(f).parents[0]).lower() in ['bg', 'background', 'backgrounds']:
            if rand_num < bg_sample:
                files.append(f)
        elif str(Path(f).parents[0]).lower() in ['obj', 'object', 'objects']:
            if rand_num < obj_sample:
                files.append(f)
        else:
            if rand_num < sample:
                files.append(f)
    return files


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

def select_image(path, filter_str='', sample=1, bg_gain=1, obj_gain=1):
    paths = glob_file(path)
    paths = select_sample(paths, filter_str=filter_str, sample=sample, bg_gain=bg_gain, obj_gain=obj_gain)
    images = select_format(paths)
    return images

def select_video(path):
    paths = glob_file(path)
    videos = select_format(paths, file_format=VID_FORMATS)
    return videos

def select_label(img_paths):
    label_files, seg_files = img2seg_labels(img_paths)  # labels
    return label_files, seg_files

def img2seg_labels(img_paths, mask_suffix='.png'):
    # Define label paths as a function of image paths
    sa, sb, sm = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep, os.sep + 'labels' + os.sep + 'masks' + os.sep
    parent =  img_paths[0].split('images')[0]
    if os.path.exists(parent+sb):
        label_path = [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]
    else:
        label_path = [None]*len(img_paths)
    if os.path.exists(parent+sm):
        mask_path = [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + mask_suffix for x in img_paths]
    else:
        mask_path = [None]*len(img_paths)
    return label_path, mask_path

def img2seg_label(img_path, mask_suffix='.png'):
    # Define label paths as a function of image paths
    sa, sb, sm = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep, os.sep + 'labels' + os.sep + 'masks' + os.sep
    parent =  img_path.split('images')[0]
    if os.path.exists(parent+sb):
        label_path = sb.join(img_path.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt'
    else:
        label_path = None
    if os.path.exists(parent+sm):
        mask_path = sb.join(img_path.rsplit(sa, 1)).rsplit('.', 1)[0] + mask_suffix
    else:
        mask_path = None
    return label_path, mask_path

def load_image(img_f):
    result = {}
    result['filename'] = img_f
    # loads 1 image from dataset index 'i', returns im, original hw, resized hw
    im = cv2.imread(img_f)  # BGR
    shape = im.shape[:2] # origin shape [height, width]
    result['ori_shape'] = shape
    result['ori_img'] = im
    assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
    assert im is not None, f'Image Not Found {img_f}'
    result['img'] = deepcopy(im)
    return result

def load_images(img_paths):
    return [load_image(img_f) for img_f in img_paths]

def load_label(result, select_class=(), prefix='', logger=None):
    img_f  = result['filename']
    label_file, seg_file = img2seg_label(img_f)  # labels
    if seg_file is not None:
        result['segment'] = cv2.imread(seg_file)
    else:
        result['segment'] = None

    # verify labels
    segments = []
    if os.path.isfile(label_file):
        with open(label_file) as f:
            l = [x.split() for x in f.read().strip().splitlines() if len(x)]
            # select category we wanted:
            if len(select_class):
                l = [x for x in l if int(x[0]) in select_class]

            if any([len(x) > 8 for x in l]):  # is segment
                classes = np.array([x[0] for x in l], dtype=np.float32)
                segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in l]  # (cls, xy1...)
                l = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
            l = np.array(l, dtype=np.float32)
        nl = len(l)
        if nl:
            # select category we wanted and sequence it from 0:
            # if l.ndim>1:
            for i, sc in enumerate(select_class):
                # l[l[..., 0]==sc][..., 0] == i # this is false, cant change l
                l[l[..., 0] == sc, 0] = i
            assert l.shape[1] == 5, f'labels require 5 columns, {l.shape[1]} columns detected'
            assert (l >= 0).all(), f'negative label values {l[l < 0]}'
            assert (l[:, 1:] <= 1).all(), f'non-normalized or out of bounds coordinates {l[:, 1:][l[:, 1:] > 1]}'
            _, i = np.unique(l, axis=0, return_index=True)
            if len(i) < nl:  # duplicate row check
                l = l[i]  # remove duplicates
                if segments:
                    segments = segments[i]
                print_log(f'{prefix}WARNING: {img_f}: {nl - len(i)} duplicate labels removed', logger)
        else:
            l = np.zeros((0, 5), dtype=np.float32)
            print_log(f'{prefix}WARNING: {img_f} label is empty', logger)
    else:
        l = np.zeros((0, 5), dtype=np.float32)
        print_log(f'{prefix}WARNING: {img_f} label is not a file, not found', logger)
    # if any(l[..., 0]>0):
    #     print(im_file, l, shape, segments)

    result['labels'] = l
    result['instance_segments'] = segments if segments else None
    return result

def load_labels(results, select_class=(), prefix='',filter_bkg=False, logger=None):
    results = [load_label(result, select_class=select_class, prefix=prefix, logger=logger) for result in results]
    if filter_bkg:
        results = [result for result in results if len(result['labels'])]
    return results

def rect_shape(results, img_size, pad, stride, batch_size):
    bi = np.floor(np.arange(len(results)) / batch_size).astype(np.int)  # batch index  #[0,0,0,...,1,1,1,..,bs]
    nb = bi[-1] + 1  # number of batches

    s = np.array([list(result['ori_shape']) for result in results]) # w,h   (hs/h) /(ws/w)
    ar = s[:, 0] / s[:, 1]  # aspect ratio
    irect = ar.argsort()
    results = [results[i] for i in irect]

    ar = ar[irect]
    # Set training image shapes and transform wh to hw
    shapes = [[1, 1]] * nb
    for i in range(nb):
        ari = ar[bi == i]
        mini, maxi = ari.min(), ari.max()
        # now ar is h/w
        if maxi <= 1:  # h < w
            shapes[i] = [1 / maxi * img_size[0], 1 / maxi * img_size[0]]  # w*h/ws
        elif mini > 1:  # w > h
            shapes[i] = [mini * img_size[1], mini * img_size[1]]  # h*ws/w
    batch_shapes = np.ceil(np.array(shapes) / stride + pad).astype(np.int) * stride
    for i, result in enumerate(results):
        result['rect_shape'] = batch_shapes[bi[i]]
        result['rectangle_stride'] = stride
    return results
# class Loader:
#     def __init__(self):

class LoadImagesAndLabels(Dataset):
    #  loads images and labels for training and validation
    def __init__(self, path,  img_size=640, batch_size=16, logger=LOGGER, augment=False, hyp=None, rect=False, image_weights=False,
                single_cls=False, stride=32, pad=0.0, prefix='',is_bgr=True, clahe=False, is_seg=False,filter_str='', select_class=(),
                 filter_bkg=False):
        if isinstance(img_size, int):
            img_size = (img_size, img_size)   # h, w
        elif len(img_size)==1:
            img_size = (img_size[0], img_size[0])
        else:
            # img_size is (w,h) ,now we convert to (h,w)
            img_size = tuple((img_size[1], img_size[0]))
        self.img_size = img_size
        self.logger =logger
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size[0] // 2, -img_size[1] // 2]
        self.stride = stride
        self.path = path

        self.is_bgr = is_bgr
        self.clahe =clahe
        # seg_path = p.parent.absolute() / 'masks'
        self.seg_flag = is_seg
        # select sample randomly
        portion = 1
        if hyp is not None:
            if 'train' in prefix:
                portion = hyp.get('train_sample_portion', 1)
            elif 'val' in prefix:
                portion = hyp.get('val_sample_portion', 1)
        self.img_files = select_image(path, filter_str=filter_str, sample=portion, bg_gain=1, obj_gain=1)
        assert self.img_files, f'{prefix}No images found'
        self.indices = range(len(self.img_files))
        results = load_images(self.img_files)
        results = load_labels(results, select_class=select_class, prefix=prefix, filter_bkg=filter_bkg, logger=logger)
        if rect:
            results = rect_shape(results, img_size, pad, stride, batch_size)

        # transform
        albumentations = Albumentations(logger)
        for result in results:
            resize(result, img_size, augment=augment)
            adapt_pad(result, auto=False, scaleup=augment) #label xywh2xyxy
            if augment:
                random_perspective(result,degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'])
                xy2wh(result)
                albumentations(result)
                augment_hsv(result['img'], hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])
                flip(result, p_ud=hyp['flipud'], p_lr=hyp['fliplr'])
            else:
                xy2wh(result)
            gray(result)
            format(result)
        self.results = results
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
            class_label = torch.from_numpy(result['class_label'])
            labels = result['labels']
            im_label = torch.zeros((len(labels),6))
            im_label[:, 0] = i
            im_label[:, 1:] = torch.from_numpy(labels)
            path = result['filename']
            shape = (result['ori_shape'],(result['resized_shape'], result['pad']))
            seg = result['segment']
            if seg is not None:
                seg = torch.from_numpy(seg)
            segs.append(seg)
            imgs.append(img)
            class_labels.append(class_label)
            im_labels.append(im_label)
            paths.append(path)
            shapes.append(shape)

        col_target = [torch.cat(class_labels,0), torch.cat(im_labels, 0), None]
        return torch.stack(imgs, 0), col_target, paths, shapes
