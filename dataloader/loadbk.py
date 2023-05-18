import glob
import os
import random
from pathlib import Path
import cv2
import math
from copy import deepcopy
from typing import Dict
# from threading import Thread
from concurrent import futures
import numpy as np
from tqdm import tqdm
from multiprocessing.pool import Pool
from functools import partial  #多参数传入函数
from utils import  segments2boxes, print_log, check_version, colorstr, xyxy2xywh, xywh2xyxy, xyn2xy, clip_label, visual_images
from dataloader.transform import adapt_pad, resize, flip, gray, format
from dataloader.augmentations import  augment_hsv, mixup, random_perspective

# Parameters
IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
VID_FORMATS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of multiprocessing threads


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
            if rand_num <= bg_sample:
                files.append(f)
        elif str(Path(f).parents[0]).lower() in ['obj', 'object', 'objects']:
            if rand_num <= obj_sample:
                files.append(f)
        else:
            if rand_num <= sample:
                files.append(f)
    return files

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

def load_image(img_f, img_size=(640,640)):
        result = {}
        result['filename'] = img_f
        # loads 1 image from dataset index 'i', returns im, original hw, resized hw
        im = cv2.imread(img_f)  # BGR
        shape = im.shape[:2] # origin shape [height, width]
        result['ori_shape'] = shape
        # result['ori_img'] = im
        assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
        assert im is not None, f'Image Not Found {img_f}'
        result['img'] = deepcopy(im)
        result['img_size'] = img_size
        result['labels'] = None
        result['segment'] = None
        result['instance_segments'] = None
        return result

def load_images(img_paths, img_size=(640,640)):
    if len(img_paths)>20:
        results = []
        with futures.ThreadPoolExecutor(NUM_THREADS) as executor:
            res = executor.map(load_image,img_paths)
        pbar = tqdm(res, total=len(img_paths), bar_format='', desc='loading_images')
        for result in pbar:
            result['img_size'] = img_size
            results.append(result)
        return results
        # for img_f in tqdm(img_paths, total=len(img_paths), bar_format='', desc='loading_images'):
            # Thread(target=load_image, args=(img_f, img_size), daemon=True).start()) # 有返回值，需要用到join 有些麻烦
        # results = []
        # with Pool(NUM_THREADS) as pool:
        #     pbar = pool.imap(load_image, img_paths)
        #     pbar = tqdm(pbar, total=len(img_paths), bar_format='', desc='loading_images')
        #     for result in pbar:
        #         result['img_size'] = img_size
        #         results.append(result)
        # return results
    else:
        return [load_image(img_f, img_size) for img_f in tqdm(img_paths, total=len(img_paths), bar_format='', desc='loading_images')]

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
            # print_log(f'{prefix}WARNING: {img_f} label is empty', logger)
    else:
        l = np.zeros((0, 5), dtype=np.float32)
        print_log(f'{prefix}WARNING: {img_f} label is not a file, not found', logger)
    # if any(l[..., 0]>0):
    #     print(im_file, l, shape, segments)

    result['labels'] = l
    result['instance_segments'] = segments if segments else None
    return result

def load_labels(results, select_class=(), prefix='',filter_bkg=False, logger=None):
    # n_rs = []
    # with Pool(NUM_THREADS) as pool:
    #     pbar = pool.imap(partial(load_label,select_class=select_class, prefix=prefix, logger=logger), results)
    #     pbar = tqdm(pbar, total=len(results), bar_format='', desc='loading_labels')
    #     for result in pbar:
    #         n_rs.append(result)
    # results = n_rs
    # # it cost more in pool start and queue, eg:100s for 5400 labels, and only 2s without pool
    results = [load_label(result, select_class=select_class, prefix=prefix, logger=logger) for result in tqdm(results, total=len(results), bar_format='', desc='loading_labels')]
    if filter_bkg:
        results = [result for result in results if len(result['labels'])]
    return results

def rect_shape(results, pad, stride, batch_size):
    img_size = results[0]['img_size']
    bi = np.floor(np.arange(len(results)) / batch_size).astype(np.int8)  # batch index  #[0,0,0,...,1,1,1,..,bs]
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
        if maxi < 1:  # h < w
            shapes[i] = [1 / maxi * img_size[0], 1 / maxi * img_size[0]]  # w*h/ws
        elif mini > 1:  # w > h
            shapes[i] = [mini * img_size[1], mini * img_size[1]]  # h*ws/w
        else:
            shapes[i] = [min(img_size), min(img_size)]
    batch_shapes = np.ceil(np.array(shapes) / stride + pad).astype(np.int32) * stride
    for i, result in enumerate(results):
        result['rect_shape'] = batch_shapes[bi[i]]
        result['rectangle_stride'] = stride
    return results

def check_label(result, func):
    if result['labels'] is not None:
        assert result['labels'].ndim==2, f'{func}, error'

def transform(result, hyp=None, augment=False, albumentations=None):
    resize(result, augment=augment)
    adapt_pad(result, auto=False, scaleup=augment)  # label xywh2xyxy
    if augment:
        random_perspective(result, degrees=hyp['degrees'],
                           translate=hyp['translate'],
                           scale=hyp['scale'],
                           shear=hyp['shear'],
                           perspective=hyp['perspective'])
        album(result, albumentations)
        augment_hsv(result['img'], hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])
        flip(result, p_ud=hyp['flipud'], p_lr=hyp['fliplr'])

    gray(result, hyp=hyp)
    format(result)
    return result

def album(result, transform=None, p=1.0):
    if transform is not None and random.random() < p:
        if result['labels'] is not None:
            new = transform(image=result['img'], bboxes=result['labels'][:, 1:],
                                 class_labels=result['labels'][:, 0])  # transformed
            result['img'], result['labels'] = new['image'], np.array(
                [[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])]).reshape(-1, 5)  # labels must be (n,5)
        else:
            result['img'] = transform(image=result['img'])  # transformed

def transforms(results, hyp=None, augment=False, logger=None):
    if augment:
        load_mosaic(results, num=len(results))
    try:
        import albumentations as A
        check_version(A.__version__, '1.0.3', hard=True)  # version requirement
        transform_list = [A.Blur(p=0.01),
                          A.MedianBlur(p=0.01),
                          A.ToGray(p=0.01),
                          A.CLAHE(p=0.01),
                          A.RandomBrightnessContrast(p=0.0),
                          A.RandomGamma(p=0.0),
                          A.ImageCompression(quality_lower=75, p=0.0)]
        compose = A.Compose(transform_list, bbox_params=A.BboxParams(format='pascal_voc', label_fields=[
            'class_labels']))  # format ['pascal_voc', 'albumentations', 'coco', 'yolo']
        print_log(colorstr('albumentations: ') + ', '.join(f'{x}' for x in compose.transforms if x.p), logger)
    except:
        compose = None

    if len(results) > 20000:
        func_p = partial(transform, hyp=hyp, augment=augment, albumentations=compose)
        new_r = []
        with Pool(NUM_THREADS) as pool:
            # pool.imap(func_p, results) # it doesnt work, as imap need iter
            for r in tqdm(pool.imap(func_p, results), total=len(results), bar_format='', desc='transforming'):
                new_r.append(r)
        return new_r
    else:
        for result in tqdm(results, total=len(results), bar_format='', desc='transforming'):
            transform(result, hyp=hyp, augment=augment, albumentations=compose)
        return results

def load_mosaic(results, num=10, more_add=4):
    # 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
    # 把seg_label看作独立于img之外的一组图

    img_h, img_w =  img_size = results[0]['img_size']
    w_num = int(math.sqrt(more_add))
    h_num = more_add // w_num
    result_l = len(results)

    for _ in range(num):
        labels_all, segments_all = [], []
        indices = random.choices(range(result_l), k=w_num * h_num)  # image indices
        img_add = np.full((img_h * h_num, img_w * w_num, 3), 114, dtype=np.uint8)  # base image
        seg_add = np.full((img_h * h_num, img_w * w_num, 3), 2, dtype=np.uint8) if results[0]['segment'] is not None else None

        new_result = {'filename':[], 'img_size':img_size, 'index':indices }
        for i, index in enumerate(indices):
            h_i, w_i = i % w_num, i//w_num
            # Load image
            result = results[index]
            img = deepcopy(result['img'])
            seg = deepcopy(result['segment'])
            segments = deepcopy(result['instance_segments'])
            labels = deepcopy(result['labels'])
            h, w = result['ori_shape']

            new_result['filename'].append(result['filename'])

            # place img in img_add
            y1 = random.randint(0, max(0, h - img_h))
            x1 = random.randint(0, max(0, w - img_w))
            h_crop = min(h-y1, img_h)
            w_crop = min(w-y1, img_w)
            by1 = h_i * img_h
            bx1 = w_i * img_w
            img_add[by1:by1+h_crop, bx1:bx1+w_crop] = img[y1:y1+h_crop, x1:x1+w_crop]

            if seg_add is not None:
                seg_add[by1:by1 + h_crop, bx1:bx1 + w_crop] = seg[y1:y1 + h_crop, x1:x1 + w_crop]


            w_bias = bx1 - x1
            h_bias = by1 - y1

            # Labels
            if labels is not None:
                labels[:, 1:] = xywh2xyxy(labels[:, 1:], w, h, w_bias, h_bias)  # normalized xywh to pixel xyxy format
                clip_label(labels[:, 1:], bx1+w_crop, by1+h_crop, bx1, by1)
                labels_all.append(labels)
            if segments is not None:
                segments = [xyn2xy(x, w, h, w_bias, h_bias) for x in segments]
                for x in segments:
                    clip_label(x, bx1 + w_crop, by1 + h_crop, bx1, by1)
                segments_all.extend(segments)

        new_pic_x = int(random.uniform(0, (w_num - 1) * img_w))
        new_pic_y = int(random.uniform(0, (h_num - 1) * img_h))
        new_img = img_add[new_pic_y: new_pic_y+img_h,new_pic_x: new_pic_x+img_w]
        if seg_add is not None:
            seg_add = seg_add[new_pic_y: new_pic_y + img_h, new_pic_x: new_pic_x + img_w]
        # Concat/clip labels

        if len(labels_all):
            labels_all = np.concatenate(labels_all, 0)
        if len(labels_all):
            labels_all[..., 1:] = xyn2xy(labels_all[..., 1:], w=1, h=1, padw=-new_pic_x, padh=-new_pic_y)
            segments_all = xyn2xy(segments_all, w=1, h=1, padw=-new_pic_x, padh=-new_pic_y)
            clip_label(labels_all[:, 1:], img_w, img_h)
            for x in segments_all:
                clip_label(x, img_w, img_h)
            # for x in (labels_all[:, 1:], *segments_all):
            #     np.clip(x[..., 0::2], 0, img_w, out=x[..., 0::2])  # clip when using random_perspective()
            #     np.clip(x[..., 1::2], 0, img_h, out=x[..., 1::2])  # clip when using random_perspective()
            # gt_result = np.zeros((labels_all.shape[0], labels_all.shape[1] + 1))
            # gt_result[:, :4] = labels_all[:, 1:]
            # gt_result[:, 4] = labels_all[:, 0]
            # visual_images(new_img, gt_result, None, './', fname=Path(new_result['filename'][0]).name)
            labels_all[..., 1:] = xyxy2xywh(labels_all[..., 1:], w=img_w, h=img_h)

        new_result['img'] = new_img
        new_result['segment'] = seg_add
        new_result['labels'] = labels_all if len(labels_all) else np.zeros((0, 5))
        new_result['instance_segments'] = segments_all if segments_all else None
        results.append(new_result)




