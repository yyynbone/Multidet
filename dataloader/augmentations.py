"""
Image augmentation functions
"""
from pathlib import Path
import math
import random
from copy import deepcopy
import cv2
import numpy as np
from dataloader.load import img2npy
from utils import  bbox_ioa, resample_segments, segment2box, xyxy2xywh, xywh2xyxy, clip_label

def augment_hsv(result, hgain=0.5, sgain=0.5, vgain=0.5):
    im = result['img']
    if im.ndim==3:  # numpy array ndim, torch.dim()
        # HSV color-space augmentation
        if hgain or sgain or vgain:
            r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
            hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
            dtype = im.dtype  # uint8

            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed

def hist_equalize(result, clahe=True, bgr=False):
    im = result['img']
    # Equalize histogram on BGR image 'im' with im.shape(n,m,3) and range 0-255
    yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV if bgr else cv2.COLOR_RGB2YUV)
    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = c.apply(yuv[:, :, 0])
    else:
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # equalize Y channel histogram
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR if bgr else cv2.COLOR_YUV2RGB)  # convert YUV image to RGB

def replicate(result):
    im = result['img']
    labels = result['labels']
    # Replicate labels and copy box to another place of the image
    h, w = im.shape[:2]
    boxes = labels[:, 1:].astype(int)
    x1, y1, x2, y2 = boxes.T
    s = ((x2 - x1) + (y2 - y1)) / 2  # side length (pixels)
    for i in s.argsort()[:round(s.size * 0.5)]:  # smallest indices
        x1b, y1b, x2b, y2b = boxes[i]
        bh, bw = y2b - y1b, x2b - x1b
        yc, xc = int(random.uniform(0, h - bh)), int(random.uniform(0, w - bw))  # offset x, y
        x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]
        im[y1a:y2a, x1a:x2a] = im[y1b:y2b, x1b:x2b]  # im4[ymin:ymax, xmin:xmax]
        labels = np.append(labels, [[labels[i, 0], x1a, y1a, x2a, y2a]], axis=0)
    result['img'] = im
    result['labels'] = labels

def random_perspective(result, degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]
    im = result['img']
    labels = result['labels']
    # # xywh to xyxy
    # if len(labels):
    #     labels[:, 1:5] = xywh2xyxy(labels[:, 1:5], w=result['img'].shape[1], h=result['img'].shape[0])

    seg = result['segment']
    segments = result['instance_segments']
    border = result.get('border', (0,0))

    height = im.shape[0] + border[0] * 2  # shape(h,w,c)
    width = im.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))
            if seg is not None:
                # seg = cv2.warpPerspective(seg, M, dsize=(width, height), borderValue=0)
                seg = cv2.warpPerspective(seg, M, dsize=(width, height), borderValue=(2,2,2))
        else:  # affine
            im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
            if seg is not None:
                # seg = cv2.warpAffine(seg, M[:2], dsize=(width, height), borderValue=0)
                seg = cv2.warpAffine(seg, M[:2], dsize=(width, height), borderValue=(2,2,2))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(im[:, :, ::-1])  # base
    # ax[1].imshow(im2[:, :, ::-1])  # warped

    # Transform label coordinates
    n = len(labels) if labels is not None else 0
    if n:
        new = np.zeros((n, 4))
        if segments is not None:  # warp segments
            segments = resample_segments(segments)  # upsample
            for i, segment in enumerate(segments):
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment
                xy = xy @ M.T  # transform
                xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]  # perspective rescale or affine

                # clip
                new[i] = segment2box(xy, width, height)

        else:  # warp boxes
            xy = np.ones((n * 4, 3))
            xy[:, :2] = labels[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=labels[:, 1:5].T * s, box2=new.T, area_thr=0.01 if segments is not None else 0.10)
        labels = labels[i]
        labels[:, 1:5] = new[i]
        if segments is not None:
            segments =  segments[i]
    # if len(labels):
    #     labels[:, 1:5] = xyxy2xywh(labels[:, 1:5], w=result['img'].shape[1], h=result['img'].shape[0], clip=True, eps=1E-3)
    result['labels'] = labels
    result['img'] = im
    result['segment'] = seg
    result['instance_segments'] = segments

def copy_paste(result, p=0.5):
    # Implement Copy-Paste augmentation https://arxiv.org/abs/2012.07177, labels as nx5 np.array(cls, xyxy)
    im = result['img']
    seg = result['segment']
    segments = result['instance_segments']
    labels = result['labels']
    n = len(segments) if segments is not None else 0
    if p and n:
        h, w, c = im.shape  # height, width, channels
        im_new = np.zeros(im.shape, np.uint8)
        if seg is not None:
            seg_new = np.zeros(seg.shape, np.uint8)
        for j in random.sample(range(n), k=round(p * n)):
            l, s = labels[j], segments[j]
            box = w - l[3], l[2], w - l[1], l[4]
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            if (ioa < 0.30).all():  # allow 30% obscuration of existing labels
                labels = np.concatenate((labels, [[l[0], *box]]), 0)
                segments.append(np.concatenate((w - s[:, 0:1], s[:, 1:2]), 1))
                cv2.drawContours(im_new, [segments[j].astype(np.int32)], -1, (255, 255, 255), cv2.FILLED)
                if seg is not None:
                    cv2.drawContours(seg_new, [segments[j].astype(np.int32)], -1, (255, 255, 255), cv2.FILLED)

        im_result = cv2.bitwise_and(src1=im, src2=im_new)
        im_result = cv2.flip(im_result, 1)  # augment segments (flip left-right)
        i = im_result > 0  # pixels to replace
        # i[:, :] = result.max(2).reshape(h, w, 1)  # act over ch
        im[i] = im_result[i]  # cv2.imwrite('debug.jpg', im)  # debug
        if seg is not None:
            seg_result = cv2.bitwise_and(src1=seg, src2=seg_new)
            seg_result = cv2.flip(seg_result, 1)
            i = seg_result > 0
            seg[i] = seg_result[i]
    result['img'] = im
    result['segment'] = seg
    result['instance_segments'] = segments
    result['labels'] = labels

def mask_label(result):
    f = Path(result['filename'])
    mask_name = f.parent / f'{f.stem}_mask.jpg'
    if mask_name.exists():
        im_paint = img2npy(mask_name)  # BGR
    else:
        im = result['img']
        max_h, max_w = im.shape[:2]
        labels = result['labels']
        big_label = deepcopy(labels)
        big_label[:, 1:] = xyxy2xywh(big_label[:, 1:], 1, 1)
        big_label[:, 3:] *= 1.1
        big_label[:, 1:] = xywh2xyxy(big_label[:, 1:])
        clip_label(big_label[:, 1:], max_w, max_h)
        inpaintMask = np.zeros(im.shape[:2], np.uint8)
        for l in big_label.astype(np.int32):
            inpaintMask[l[2]:l[4], l[1]:l[3]] = 1
        im_paint = cv2.inpaint(im, inpaintMask, inpaintRadius=2, flags=cv2.INPAINT_TELEA)
        cv2.imwrite(str(mask_name), im_paint)
    result['labels'] = np.zeros((0,5))
    result['img'] = im_paint
    result['filename'] = str(mask_name)

    # im = result['img']
    # max_h, max_w = im.shape[:2]
    # labels = result['labels']
    # big_label = deepcopy(labels)
    # big_label[:, 1:] = xyxy2xywh(big_label[:, 1:], 1, 1)
    # big_label[:, 3:] *= 1.5
    # big_label[:, 1:] = xywh2xyxy(big_label[:, 1:])
    # clip_label(big_label[:, 1:], max_w, max_h)
    # # inpaintMask = np.zeros(im.shape[:2], np.uint8)
    # for  l in labels.astype(np.int32):
    #     im[l[2]:l[4], l[1]:l[3]] = 0
    # for l, b_l in zip(labels.astype(np.int32), big_label.astype(np.int32)):
    #     big_array = im[b_l[2]:b_l[4], b_l[1]:b_l[3]]
    #     new = big_array[big_array > 0].flatten()
    #     w, h = l[3]-l[1], l[4]-l[2]
    #     a = np.random.choice(new, h*w*3)
    #     im[l[2]:l[4], l[1]:l[3]] = a.reshape(h, w, 3)
    #     # np.random.shuffle(im[l[2]:l[4], l[1]:l[3]]) # im[l[2]:l[4], l[1]:l[3]].mean()
    #     # im[l[2]:l[4], l[1]:l[3]] =
    #     # inpaintMask[l[2]:l[4], l[1]:l[3]] = 1
    # cv2.imwrite('local_inpaint.jpg', im)
    # res1 = cv2.inpaint(im, inpaintMask, inpaintRadius=10, flags=cv2.INPAINT_NS)
    # res2 = cv2.inpaint(im, inpaintMask, inpaintRadius=2, flags=cv2.INPAINT_TELEA)
    # cv2.imwrite('shuffle.jpg', im)
    # cv2.imwrite('ns_inpaint.jpg', res1)
    # cv2.imwrite('fast_inpaint.jpg', res2)
    # im = result['img']
    # max_h, max_w = im.shape[:2]
    # labels = result['labels']
    # r = 0.2
    # resized_shape = (int(max_w * r), int(max_h * r))
    # new_im = cv2.resize(im, resized_shape, cv2.INTER_LINEAR)
    # labels[:, 1:] = xyxy2xywh(labels[:, 1:], 1/r, 1/r)
    # labels[:, 3:] *= 1.2
    # labels[:, 1:] = xywh2xyxy(labels[:, 1:])
    # labels = labels.astype(np.int32)
    # clip_label(labels[:, 1:], resized_shape[0], resized_shape[1])
    # cv2.imwrite('resize_zero.jpg', new_im)
    # for l in labels:
    #     new_im[l[2]:l[4], l[1]:l[3]] = new_im[l[2]:l[4], l[1]:l[3]].mean()
    # cv2.imwrite('resize_zero_mean.jpg', new_im)
    # kernel = np.ones((3, 3))/9
    # new_im = cv2.filter2D(new_im, -1, kernel)
    # cv2.imwrite('resize_zero_mean_filter.jpg', new_im)
    # new_im = cv2.resize(new_im,(max_w, max_h) , cv2.INTER_LINEAR)
    # cv2.imwrite('resize_zero_mean_filter_scale.jpg', new_im)
    # for l in labels:
    #     im[l[2]:l[4], l[1]:l[3]] = im[l[2]:l[4], l[1]:l[3]].mean()

    # kernel = np.ones((10, 10), np.uint8)
    # erose_img = cv2.erode(im, kernel, iterations = 1)
    # cv2.imwrite('erode.jpg', erose_img)
    #
    # erose_img = cv2.dilate(im, kernel, iterations = 1)
    # cv2.imwrite('dilate.jpg', erose_img)
    # for l in labels:
    #     im[l[2]:l[4], l[1]:l[3]] = erose_img [l[2]:l[4], l[1]:l[3]]
    # cv2.imwrite('now.jpg', im)
    # im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
    # cv2.imwrite('dilate_erode.jpg', im)
    # result['labels'] = np.zeros((0,5))
    # result['img'] = im


def cutout(result, p=0.5):
    # Applies image cutout augmentation https://arxiv.org/abs/1708.04552
    im = result['img']
    labels = result['labels']
    if random.random() < p:
        h, w = im.shape[:2]
        scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
        for s in scales:
            mask_h = random.randint(1, int(h * s))  # create random masks
            mask_w = random.randint(1, int(w * s))

            # box
            xmin = max(0, random.randint(0, w) - mask_w // 2)
            ymin = max(0, random.randint(0, h) - mask_h // 2)
            xmax = min(w, xmin + mask_w)
            ymax = min(h, ymin + mask_h)

            # apply random color mask
            im[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

            # return unobscured labels
            if len(labels) and s > 0.03:
                box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
                ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
                labels = labels[ioa < 0.60]  # remove >60% obscured labels
    result['img'] = im
    result['labels'] = labels

def mixup(result1, result2):
    # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
    im1, labels1, seg1 = result1['img'], result1['labels'], result1['segment']
    im2, labels2, seg2 = result2['img'], result2['labels'], result2['segment']
    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
    im = (im1 * r + im2 * (1 - r)).astype(np.uint8)
    if seg1 is not None:
        seg = (seg1 * r + seg2 * (1 - r)).astype(np.uint8)
    labels = np.concatenate((labels1, labels2), 0)
    result = {}
    result['img'] = im
    result['labels'] = labels
    result['mixup'] = [result1, result2]
    return result

def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates

