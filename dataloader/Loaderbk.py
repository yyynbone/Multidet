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

from dataloader.check_loadbk import IMG_FORMATS, VID_FORMATS, img2label_paths, img2seglabel_paths, attach_labels, load_mosaic, load_image

from dataloader.augmentationsbk import Albumentations, augment_hsv, letterbox, mixup, random_perspective, Clahe

from utils import (set_logging, xywh2xyxy, xyxy2xywh, clean_str)

LOGGER = set_logging(__name__)

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

class LoadImages:
    # YOLOv5 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`
    def __init__(self, path, img_size=640, stride=32, auto=True, is_bgr=True):
        p = str(Path(path).resolve())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
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
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'

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

        # Padded resize
        img,seg_img = letterbox((img0,None), self.img_size, stride=self.stride, auto=self.auto)[0]

        # Convert
        if self.is_bgr:
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img = np.ascontiguousarray(img)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img[None]

        return path, img, img0, self.cap, s

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files

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
        self.albumentations = Albumentations(logger) if augment else None
        self.is_bgr = is_bgr
        self.clahe =clahe
        # seg_path = p.parent.absolute() / 'masks'
        self.seg_flag = is_seg

        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise Exception(f'{prefix}{p} does not exist')
            self.img_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert self.img_files, f'{prefix}No images found'
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}\n')

        self.img_files = sorted(self.img_files, key=lambda x:x[-5:])
        # select sample randomly
        if hyp is not None:
            if 'train' in prefix:
                portion = hyp.get('train_sample_portion', 1)
            elif 'val' in prefix:
                portion = hyp.get('val_sample_portion', 1)
            else:
                portion = 1
            select_num  = int(len(self.img_files) * portion)
            self.img_files = self.img_files[:select_num]

        # select image data which contains filter_str
        if filter_str:
            self.img_files = [im_f for im_f in self.img_files if filter_str in im_f]

        # Check cache
        self.label_files = img2label_paths(self.img_files)  # labels
        if self.seg_flag:
            self.seg_files = img2seglabel_paths(self.img_files) #seg_labels
        else:
            self.seg_files = [None]*len(self.img_files)

        at_labels = attach_labels(self.img_files, self.label_files, prefix, select_class=select_class, logger=logger)
        # exit()
        # Display cache
        nf, nm, ne, nc, n = at_labels.pop('results')  # found, missing, empty, corrupted, total

        assert nf > 0 or not augment, f'{prefix}No labels in {p.parent}. Can not train without labels.'


        # Read cache
        labels, shapes, self.segments = zip(*at_labels.values())

        # filter background img and labels, contains self.img_files, self.label_files, self.seg_files, at_labels(labels, shapes, self.segments),
        if filter_bkg:
            idx = [i for i,l in enumerate(labels) if len(l)]
            self.img_files = np.array(self.img_files)[idx].tolist()
            self.label_files = np.array(self.label_files)[idx].tolist()
            self.seg_files = np.array(self.seg_files)[idx].tolist()
            shapes = np.array(shapes)[idx]
            labels = np.array(list(labels))[idx]
            if self.segments:
                self.segments = np.array(self.segments)[idx]

        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        # self.shapes = np.array(self.ori_img_size, dtype=np.float64)


        n = len(self.img_files)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(np.int8)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)

        # Update labels
        include_class = []  # filter labels to include only these classes (optional)
        include_class_array = np.array(include_class).reshape(1, -1)
        for i, (label, segment) in enumerate(zip(self.labels, self.segments)):
            if include_class:
                j = (label[:, 0:1] == include_class_array).any(1)
                self.labels[i] = label[j]
                if segment:
                    self.segments[i] = segment[j]
            if single_cls:  # single-class training, merge all classes into 0
                self.labels[i][:, 0] = 0
                if segment:
                    self.segments[i][:, 0] = 0
        '''
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # w,h
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            if self.seg_flag:
                self.seg_files = [self.seg_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]   # (h/w,1)
                elif mini > 1:
                    shapes[i] = [1, 1 / mini] # (1,w/h)
            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int8) * stride # hs,ws = img_size,  if h>w, (hs,w*ws/h) else (hs*h/w,ws)
        '''
        # Rectangular Training, 即将不规则的图形，规则为正方形状， 如 （600，400） 通过上下补黑边，
        # 将输入图像转为（600，600）， 其中batch shape 的（600,600） 在此处计算得出
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # w,h   (hs/h) /(ws/w)
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            if self.seg_flag:
                self.seg_files = [self.seg_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]
            # Set training image shapes and transform wh to hw
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                # now ar is h/w
                if maxi < 1:    # h < w
                    shapes[i] = [ 1/maxi * img_size[0], 1/maxi * img_size[0] ]# w*h/ws
                elif mini > 1:  # w > h
                    shapes[i] = [ mini * img_size[1], mini * img_size[1]] # h*ws/w
            self.batch_shapes = np.ceil(np.array(shapes)/ stride + pad).astype(np.int8) * stride
            # now, hs,ws = img_size,  if h>w, (hs, h*ws/w) else (hs*w/h, ws)
            # self.batch_shapes

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs, self.img_npy, self.seg_labels = [None] * n, [None] * n, [None] * n



    def __len__(self):
        return len(self.img_files)


    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights
        # print('now indices len is ', len(self.indices), 'get index',index)

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        if mosaic:
            # Load mosaic
            # 同时读取和处理了img 和 seg
            # img, labels = load_mosaic(self, index)
            # print('----------------------load_mosaic----------------------')
            (img, seg_label), labels = load_mosaic(self, index)
            shapes = None
            # print('img after mosaic:',img)
            # print('seg_label after mosaic:',seg_label)
            # exit()
            # MixUp augmentation
            if random.random() < hyp['mixup']:
                # 同时mixup img和seg
                # img, labels = mixup(img, labels, *load_mosaic(self, random.randint(0, self.n - 1)))
                (img, seg_label), labels = mixup((img, seg_label), labels, *load_mosaic(self, random.randint(0, self.n - 1)))

        else:
            # Load image
            # img, (h0, w0), (h, w) = load_image(self, index)
            # print('----------------------load_directly----------------------')
            img, seg_label, (h0, w0), (h, w) = load_image(self, index)
            # print("in getitem not mosaic", (h, w)) # origin(1080,1024) , (h,w)=(675,640)
            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape, (h,w)
            # print("in getitem not mosaic before letterbox", shape) # (1088,640)
            # shape = (544, 960)
            # img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            # print("in getitem not mosaic before letterbox", shape)
            (img, seg_label), ratio, pad = letterbox((img, seg_label), shape, auto=False, scaleup=self.augment)
            
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy()
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywh2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])
            # print("in getitem not mosaic", img.shape)
            if self.augment:
                (img, seg_label), labels = random_perspective((img, seg_label), labels,
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'])
        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)

        if self.augment:
            # Albumentations
            img, labels = self.albumentations(img, labels)
            nl = len(labels)  # update after albumentations

            if self.is_bgr:
                # HSV color-space
                augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # Flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if self.seg_flag:
                    seg_label = np.flipud(seg_label)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            # Flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if self.seg_flag:
                    seg_label = np.fliplr(seg_label)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

            # Cutouts
            # labels = cutout(img, labels, p=0.5)

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)


        if not self.is_bgr:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            is_clahe = self.clahe and random.random() < hyp['clahe']
            if is_clahe:
                img = Clahe(img)
            img = img[...,None]

        # Convert
        img = img.transpose((2, 0, 1))  # HWC to CHW
        img = np.ascontiguousarray(img[::-1])  # BGR to RGB
        if self.seg_flag:
            # 现在读取的是3通道的图 shape是[1280, 640, 3]
            seg_img = seg_label.copy()
            if not self.is_bgr:
                seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY)[...,None]
            seg_img = torch.from_numpy(seg_img).float()

        else:
            seg_img = None

        class_label = torch.ones((1,1))
        if nl:
            class_label = torch.zeros((1,1))
        target = (class_label, labels_out, seg_img)
        return torch.from_numpy(img), target, self.img_files[index], shapes

    @staticmethod
    def collate_fn(batch):
        img, target, paths, shapes = zip(*batch)  # transposed
        class_label, label, seg = zip(*target)
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        col_target = [torch.cat(class_label,0), torch.cat(label, 0), None]
        return torch.stack(img, 0), col_target, paths, shapes

    @staticmethod
    def collate_fn4(batch):
        img, target, path, shapes = zip(*batch)  # transposed
        class_label, label, seg = zip(*target)
        n = len(shapes) // 4
        img4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]

        ho = torch.tensor([[0.0, 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0.0, 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, 0.5, 0.5, 0.5, 0.5]])  # scale
        for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
            i *= 4
            if random.random() < 0.5:
                im = F.interpolate(img[i].unsqueeze(0).float(), scale_factor=2.0, mode='bilinear', align_corners=False)[
                    0].type(img[i].type())
                l = label[i]
            else:
                im = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1)), 2)
                l = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            img4.append(im)
            label4.append(l)

        for i, l in enumerate(label4):
            l[:, 0] = i  # add target image index for build_targets()

        col_target = (torch.cat(class_label, 0), torch.cat(label4, 0), torch.cat(seg, 0))
        return torch.stack(img4, 0), col_target, path4, shapes4

class LoadStreams:
    # streamloader, i.e. `python detect.py --source 'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP streams`
    def __init__(self, logger=None, sources='streams.txt', img_size=640, stride=32, auto=True):
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride
        self.logger =logger

        if os.path.isfile(sources):
            with open(sources) as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        n = len(sources)
        self.imgs, self.fps, self.frames, self.threads = [None] * n, [0] * n, [0] * n, [None] * n
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        self.auto = auto
        for i, s in enumerate(sources):  # index, source
            # Start thread to read frames from video stream
            st = f'{i + 1}/{n}: {s}... '
            # if 'youtube.com/' in s or 'youtu.be/' in s:  # if source is YouTube video
            #     check_requirements(('pafy', 'youtube_dl'))
            #     import pafy
            #     s = pafy.new(s).getbest(preftype="mp4").url  # YouTube URL
            s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
            cap = cv2.VideoCapture(s)
            assert cap.isOpened(), f'{st}Failed to open {s}'
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps[i] = max(cap.get(cv2.CAP_PROP_FPS) % 100, 0) or 30.0  # 30 FPS fallback
            self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')  # infinite stream fallback

            _, self.imgs[i] = cap.read()  # guarantee first frame
            self.threads[i] = Thread(target=self.update, args=([i, cap, s]), daemon=True)
            self.logger.info(f"{st} Success ({self.frames[i]} frames {w}x{h} at {self.fps[i]:.2f} FPS)")
            self.threads[i].start()
        self.logger.info('')  # newline

        # check for common shapes
        s = np.stack([letterbox(x, self.img_size, stride=self.stride, auto=self.auto)[0].shape for x in self.imgs])
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            self.logger.warning('WARNING: Stream shapes differ. For optimal performance supply similarly-shaped streams.')

    def update(self, i, cap, stream):
        # Read stream `i` frames in daemon thread
        n, f, read = 0, self.frames[i], 1  # frame number, frame array, inference every 'read' frame
        while cap.isOpened() and n < f:
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n % read == 0:
                success, im = cap.retrieve()
                if success:
                    self.imgs[i] = im
                else:
                    LOGGER.warning('WARNING: Video stream unresponsive, please check your IP camera connection.')
                    self.imgs[i] = np.zeros_like(self.imgs[i])
                    cap.open(stream)  # re-open stream if signal was lost
            time.sleep(1 / self.fps[i])  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img0 = self.imgs.copy()
        img = [letterbox(x, self.img_size, stride=self.stride, auto=self.rect and self.auto)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None, ''

    def __len__(self):
        return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years
