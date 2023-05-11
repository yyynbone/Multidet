"""
Image transform functions,  edit by Huang Zhihua
"""
from multiprocessing.pool import Pool
import albumentations as A
from functools import partial  #多参数传入函数
from utils import  check_version, colorstr
from dataloader.augmentations import  *
from dataloader.load import *

def adapt_pad(result, pad_color=(114, 114, 114),seg_pad_color=(0, 0, 0), auto=True, scaleFill=False, scaleup=True, rectangle_stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    # the input of new_shape is (h,w)
    im = result['img']
    seg = result['segment']
    new_shape = result.get('rect_shape', result['img_size']) # rect batch shape or imgsize
    rectangle_stride = result.get('rectangle_stride', rectangle_stride)
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
            # w, h = ratio[1] * result['resized_shape'][1], ratio[0] * result['resized_shape'][0]
            # cxywh to cxywh
            # labels[:, 1] += dw / new_shape[1]
            # labels[:, 2] += dh / new_shape[2]
            labels[:, 1:] = xyn2xy(labels[:, 1:], ratio[1], ratio[0], padw=dw, padh=dh)
            result['labels'] = labels
            segments = result['instance_segments']
            if segments is not None:
                segments = [xyn2xy(x, ratio[1], ratio[0], dw, dh) for x in segments]
                result['instance_segments'] = segments

def resize(result, augment=False):
    h, w = result['img'].shape[:2]
    # h, w = result['img_shape']
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
        labels = result.get('labels', None)
        if labels is not None:
            if labels.size:
                labels[:, 1:] = xyn2xy(labels[:, 1:], r, r)

                result['labels'] = labels
                segments = result['instance_segments']
                if segments is not None:
                    segments = [xyn2xy(x, r, r, 0, 0) for x in segments]
                    result['instance_segments'] = segments

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
        if labels is not None:
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

def gray(result, clahe_p=0):
    img = cv2.cvtColor(result['img'], cv2.COLOR_BGR2GRAY)
    if random.random() < clahe_p:
        img = clahe(img)
    result['img'] = img[...,None]#(h, w, 1)
    if result['segment'] is not None:
        result['segment'] = cv2.cvtColor(result['segment'], cv2.COLOR_BGR2GRAY)[..., None]

# def format(result):
#     assert not isinstance(result['labels'], list), f'error, format, {result}'
#     labels = result['labels']
#     result['class_label'] = None
#     if labels is not None:
#         # result['wh'] = np.vstack([labels[:, 3]-labels[:, 1], labels[:, 4]-labels[:, 2]]).T
#         result['class_label'] = np.zeros((1, 1))
#         # labels to cxcywh
#         labels[:, 1:5] = xyxy2xywh(labels[:, 1:5], w=result['img'].shape[1], h=result['img'].shape[0], clip=True,
#                                    eps=1E-3)
#         if labels.shape[0]:
#             result['class_label'] += 1
#         result['labels'] = labels
#
#     # Convert
#     img = result['img']
#     img = img.transpose((2, 0, 1))  # HWC to CHW
#     img = np.ascontiguousarray(img[::-1])  # BGR to RGB
#     result['img'] = img

def dict2eval(objs, is_class=None):
    if isinstance(objs, list):
        return [dict2eval(obj, is_class=is_class) for obj in objs]
    else:
        funcs = []
        for k, args in objs.items():
            if is_class is not None:
                funcs.append(getattr(is_class, k)(**args))
            else:
                if args is None:
                    funcs.append(eval(k))
                else:
                    funcs.append(partial(eval(k), **args))

        return funcs[0] if len(funcs)==1 else funcs

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


class Transfrom():
    def __init__(self, trans_dict, logger=None):
        # if augment:
        #     load_mosaic(results, num=len(results))

        if 'resize' in trans_dict['transform'].keys():
            if trans_dict['transform']['resize'] is not None:
                trans_dict['transform']['resize']['augment'] = trans_dict.get('augment', False)
            else:
                trans_dict['transform']['resize'] = {"augment": trans_dict.get('augment', False)}

        # if 'gray' in trans_dict['transform'].keys():
        #     if trans_dict['transform']['gray'] is not None:
        #         trans_dict['transform']['gray']['to_gray'] = trans_dict['to_gray']
        #     else:
        #         trans_dict['transform']['gray'] = {"to_gray": trans_dict['to_gray']}

        check_version(A.__version__, '1.0.3', hard=True)  # version requirement
        albums = trans_dict.get('albumentations', None)
        if albums is not None:
            for para, value in albums.items():
                albums[para] = dict2eval(value, A)

            compose = A.Compose(**albums)  # format ['pascal_voc', 'albumentations', 'coco', 'yolo']
            print_log(colorstr('albumentations: ') + ', '.join(f'{x}' for x in compose.transforms if x.p), logger)
        else:
            compose = None
        if trans_dict.get('transform').get('album'):
            trans_dict['transform']['album'] = {'transform': compose}
        self.transform = dict2eval(trans_dict['transform'])

    def __call__(self, *args, **kwargs):
        for t in self.transform:
            t(*args)
