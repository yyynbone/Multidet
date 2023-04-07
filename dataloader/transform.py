"""
Image transform functions
"""

import random

import cv2
import numpy as np

from utils import  xywh2xyxy,xyxy2xywh, xyn2xy

def adapt_pad(result, pad_color=(114, 114, 114),seg_pad_color=(0, 0, 0), auto=True, scaleFill=False, scaleup=True):
    # Resize and pad image while meeting stride-multiple constraints
    # the input of new_shape is (h,w)
    im = result['img']
    seg = result['segment']
    new_shape = result.get('rect_shape', result['img_size']) # rect batch shape or imgsize
    rectangle_stride = result.get('rectangle_stride', 32)
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)  # h,w

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, rectangle_stride), np.mod(dh, rectangle_stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[0] / shape[0], new_shape[1] / shape[1]  #  height, width ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        if seg is not None:
            seg = cv2.resize(seg, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)  # add border
    # seg = cv2.copyMakeBorder(seg, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    if seg is not None:
        seg = cv2.copyMakeBorder(seg, top, bottom, left, right, cv2.BORDER_CONSTANT, value=seg_pad_color)
    result['img'] = im
    result['segment'] = seg
    labels = result.get('labels', None)
    result['pad'] = (dw, dh)
    if labels is not None:
        if labels.size:  # normalized xywh to pixel xyxy format
            w, h = ratio[1] * result['resized_shape'][1], ratio[0] * result['resized_shape'][0]
            # cxywh to cxywh
            # labels[:, 1] += dw / new_shape[1]
            # labels[:, 2] += dh / new_shape[2]
            labels[:, 1:] = xywh2xyxy(labels[:, 1:], w, h, padw=dw, padh=dh)
            result['labels'] = labels
            segments = result['instance_segments']
            if segments is not None:
                segments = [xyn2xy(x, w, h, dw, dh) for x in segments]
                result['instance_segments'] = segments

def resize(result, augment=False):
    h, w = result['img'].shape[:2]
    r = min(result['img_size'][0] / h, result['img_size'][1] / w)  # ratio
    result['resized_shape'] = (h, w)
    if r != 1:  # if sizes are not equal
        interpolation = cv2.INTER_AREA if r < 1 and not augment else cv2.INTER_LINEAR
        result['resized_shape'] = (int(w * r), int(h * r))
        result['img'] = cv2.resize(result['img'], result['resized_shape'],
                        interpolation=interpolation)
        if result['segment'] is not None:
            result['segment'] = cv2.resize(result['segment'], result['resized_shape'],
                                       interpolation=interpolation)

def flip(result, p_ud=0.5, p_lr=0.5):
    img = result['img']
    seg = result['segment']
    segments = result['instance_segments']
    labels = result['labels']
    h, w = img.shape[:2]
    if random.random() < p_ud:
        img = np.flipud(img)
        if seg is not None:
            seg = np.flipud(seg)
        if len(labels):
            labels[:, 2], labels[:, 4] = h - labels[:, 4], h - labels[:, 2]
            # segments

    # Flip left-right
    if random.random() < p_lr:
        img = np.fliplr(img)
        if seg is not None:
            seg = np.fliplr(seg)
        if labels is not None:
            labels[:, 1], labels[:, 3] = w - labels[:, 3], w - labels[:, 1]

    result['img'] = img
    result['segment'] = seg
    result['instance_segments'] = segments
    result['labels'] = labels

def clahe(img, clipLimit=2.0, tileGridSize=(8, 8)):
    if len(img.shape)!=2:
        print('CLAHE must be seg image, so we transfor it to seg')
            #transpose to c, h, w
        transpose_flag = False
        if img.shape[0]>3:
            img = img.transpose(1,2,0)
            transpose_flag = True
        img_channel = img.shape[0]
        if img_channel==3:
            img = img[:1, ...]*0.299  + img[1:2, ...]*0.587 + img[2:, ...]*0.114  # 22bridge 2airport
        img4 = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        img = img4.apply(img[0])
        img = img[None].repeat(img_channel,axis=0)
        if transpose_flag:
            img = img.transpose(2, 0, 1)
    else:
        img4 = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        img = img4.apply(img)

    return img

def result_clahe(result, clipLimit=2.0, tileGridSize=(8, 8), p=0.5):
    if random.random() < p:
        img = result['img'][..., 0]
        img = clahe(img, clipLimit=clipLimit, tileGridSize=tileGridSize)
        result['img'] = img[None].repeat(3, axis=0)

def gray(result, hyp=None):
    if hyp['to_gray']:
        img = cv2.cvtColor(result['img'], cv2.COLOR_BGR2GRAY)
        if random.random() < hyp['clahe']:
            img = clahe(img)
        result['img'] = img[...,None]#(h, w, 1)
        if result['segment'] is not None:
            result['segment'] = cv2.cvtColor(result['segment'], cv2.COLOR_BGR2GRAY)[..., None]

def format(result):

    labels = result['labels']
    result['class_label'] = None
    if labels is not None:
        result['class_label'] = np.zeros((1, 1))
        # labels to cxcywh
        labels[:, 1:5] = xyxy2xywh(labels[:, 1:5], w=result['img'].shape[1], h=result['img'].shape[0], clip=True,
                                   eps=1E-3)
        if labels.shape[0]:
            result['class_label'] += 1
        result['labels'] = labels

    # Convert
    img = result['img']
    img = img.transpose((2, 0, 1))  # HWC to CHW
    img = np.ascontiguousarray(img[::-1])  # BGR to RGB
    result['img'] = img
