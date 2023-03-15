from copy import copy
import torch
import numpy as np
import math
from pathlib import Path
import warnings
import cv2
import matplotlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from threading import Thread
from PIL import Image, ImageDraw
from utils.label_process import xywh2xyxy, xyxy2xywh, clip_coords
from utils.logger import set_logging, Colors
from utils.mix_utils import  check_font, is_ascii, is_chinese, mkdir, increment_path

LOGGER = set_logging(__name__)

colors = Colors()  # create instance for 'from utils.plots import colors'

class Annotator:
    # YOLOv5 Annotator for train/val mosaics and jpgs and detect/hub inference annotations
    def __init__(self, im, line_width=None, font_size=None, font='Arial.ttf', pil=False, example='abc'):
        assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to Annotator() input images.'
        self.pil = pil or not is_ascii(example) or is_chinese(example)
        if self.pil:  # use PIL
            self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
            self.draw = ImageDraw.Draw(self.im)
            self.font = check_font(font='Arial.Unicode.ttf' if is_chinese(example) else font,
                                   size=font_size or max(round(sum(self.im.size) / 2 * 0.035), 12))
        else:  # use cv2
            self.im = im
        self.lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)  # line width
    def mark(self, box, color=2):
        if self.pil:
            img = np.asarray(self.im)
            for i in range(3):
                if i == color:
                    img[box[1]:box[3], box[0]:box[2], i] = 0
                else:
                    img[box[1]:box[3], box[0]:box[2], i] = img[box[1]:box[3], box[0]:box[2], i]//2 + 127 #255//2=127
            self.im = Image.fromarray(img)
            self.draw = ImageDraw.Draw(self.im)

        else:
            for i in range(3):
                if i == 2 - color:
                    self.im[box[1]:box[3], box[0]:box[2], i] = 0
                else:
                    self.im[box[1]:box[3], box[0]:box[2], i] = self.im[box[1]:box[3], box[0]:box[2], i]//2 + 127 #255//2=127

    def box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        # Add one xyxy box to image with label
        if self.pil or not is_ascii(label):
            self.draw.rectangle(box, width=self.lw, outline=color)  # box
            if label:
                w, h = self.font.getsize(label)  # text width, height
                outside = box[1] - h >= 0  # label fits outside box
                self.draw.rectangle([box[0],
                                     box[1] - h if outside else box[1],
                                     box[0] + w + 1,
                                     box[1] + 1 if outside else box[1] + h + 1], fill=color)
                # self.draw.text((box[0], box[1]), label, fill=txt_color, font=self.font, anchor='ls')  # for PIL>8.0
                self.draw.text((box[0], box[1] - h if outside else box[1]), label, fill=txt_color, font=self.font)
        else:  # cv2
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
            if label:
                tf = max(self.lw - 1, 1)  # font thickness
                w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[0]  # text width, height
                outside = p1[1] - h - 3 >= 0  # label fits outside box
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(self.im, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, self.lw / 3, txt_color,
                            thickness=tf, lineType=cv2.LINE_AA)

    def rectangle(self, xy, fill=None, outline=None, width=1):
        # Add rectangle to image (PIL-only)
        self.draw.rectangle(xy, fill, outline, width)

    def text(self, xy, text, txt_color=(255, 255, 255)):
        # Add text to image (PIL-only)
        w, h = self.font.getsize(text)  # text width, height
        self.draw.text((xy[0], xy[1] - h + 1), text, fill=txt_color, font=self.font)

    def result(self):
        # Return annotated image as array
        return np.asarray(self.im)

def plot_pr_curve(px, py, ap, save_dir='pr_curve.png', names=()):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
    plt.close()

def plot_mc_curve(px, py, save_dir='mc_curve.png', names=(), xlabel='Confidence', ylabel='Metric'):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')  # plot(confidence, metric)

    y = py.mean(0)
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
    plt.close()

def plot_confusionmatrics(matrix, nc, normalize=True, save_dir='', names=()):
    # print('now we are in confusion matrix plot')
    try:
        import seaborn as sn

        array = matrix / ((matrix.sum(0).reshape(1, -1) + 1E-6) if normalize else 1)  # normalize columns
        array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

        fig = plt.figure(figsize=(12, 9), tight_layout=True)
        sn.set(font_scale=1.0 if nc < 50 else 0.8)  # for label size
        labels = (0 < len(names) < 99) and len(names) == nc  # apply names to ticklabels
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
            sn.heatmap(array, annot= nc < 30, annot_kws={"size": 8}, cmap='Blues', fmt='.2f', square=True,
                       xticklabels=names + ['background FP'] if labels else "auto",
                       yticklabels=names + ['background FN'] if labels else "auto").set_facecolor((1, 1, 1))
        fig.axes[0].set_xlabel('True')
        fig.axes[0].set_ylabel('Predicted')
        fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
        plt.close()
    except Exception as e:
        print(f'WARNING: ConfusionMatrix plot failure: {e}')
    # print('now we plot done')

def feature_visualization(x, module_type, stage, n=32, save_dir=Path('runs/detect/exp')):
    """
    x:              Features to be visualized
    module_type:    Module type
    stage:          Module stage within model
    n:              Maximum number of feature maps to plot
    save_dir:       Directory to save results
    """
    if 'Detect' and 'Classify' not in module_type:
        batch, channels, height, width = x.shape  # batch, channels, height, width
        if height > 1 and width > 1:
            f = save_dir / f"stage{stage}_{module_type.split('.')[-1]}_features.png"  # filename

            blocks = torch.chunk(x[0].cpu(), channels, dim=0)  # select batch index 0, block by channels
            n = min(n, channels)  # number of plots
            fig, ax = plt.subplots(math.ceil(n / 8), 8, tight_layout=True)  # 8 rows x n/8 cols
            ax = ax.ravel()
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            for i in range(n):
                ax[i].imshow(blocks[i].squeeze())  # cmap='gray'
                ax[i].axis('off')

            print(f'Saving {f}... ({n}/{channels})')
            plt.savefig(f, dpi=300, bbox_inches='tight')
            plt.close()
            np.save(str(f.with_suffix('.npy')), x[0].cpu().numpy())  # npy save

def plot_images(images, targets, paths=None, fname='images.jpg', names=None, max_size=1920, max_subplots=16):
    # Plot image grid with labels
    if isinstance(targets,tuple):
        targets,seg_tar = targets  # detect_out  seg_out(bs,h,w)
        seg_mask = torch.nn.functional.one_hot(seg_tar)
        if isinstance(seg_mask,torch.Tensor):
            seg_mask =seg_mask.cpu().float().numpy()*255
    else:
        seg_mask=None
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if np.max(images[0]) <= 1:
        images *= 255  # de-normalise (optional)
    bs, _, h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs ** 0.5)  # number of subplots (square)

    # Build Image
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
    for i, im in enumerate(images):
        if i == max_subplots:  # if last batch has fewer images than we expect
            break
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        im = im.transpose(1, 2, 0)
        # add seg mask to image
        if seg_mask  is not None:
            im = seg_mask[i] * 0.3 + im * 0.7
            im = im.astype(int)
            im = np.where(im>255,255,im).astype(np.uint8)

        mosaic[y:y + h, x:x + w, :] = im

    # Resize (optional)
    scale = max_size / ns / max(h, w)
    if scale < 1:
        h = math.ceil(scale * h)
        w = math.ceil(scale * w)
        mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))

    # Annotate
    fs = int((h + w) * ns * 0.01)  # font size
    annotator = Annotator(mosaic, line_width=round(fs / 10), font_size=fs, pil=True)
    for i in range(i + 1):
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        annotator.rectangle([x, y, x + w, y + h], None, (255, 255, 255), width=2)  # borders
        if paths:
            annotator.text((x + 5, y + 5 + h), text=Path(paths[i]).name[:40].replace(":", "_"), txt_color=(220, 220, 220))  # filenames
        if len(targets) > 0:
            ti = targets[targets[:, 0] == i]  # image targets
            boxes = xywh2xyxy(ti[:, 2:6]).T
            classes = ti[:, 1].astype('int')
            labels = ti.shape[1] == 6  # labels if no conf column
            conf = None if labels else ti[:, 6]  # check for confidence presence (label vs pred)

            if boxes.shape[1]:
                if boxes.max() <= 1.01:  # if normalized with tolerance 0.01
                    boxes[[0, 2]] *= w  # scale to pixels
                    boxes[[1, 3]] *= h
                elif scale < 1:  # absolute coords need scale if image scales
                    boxes *= scale
            boxes[[0, 2]] += x
            boxes[[1, 3]] += y
            for j, box in enumerate(boxes.T.tolist()):
                cls = classes[j]
                color = colors(cls)
                cls = names[cls] if names else cls
                if labels or conf[j] > 0.25:  # 0.25 conf thresh
                    if seg_mask is None:
                        label = f'{cls}' if labels else f'{cls} {conf[j]:.1f}'
                    else:
                        label=''
                    annotator.box_label(box, label, color=color)

    annotator.im.save(fname)  # save

def visual_images(im, target, pred, paths=None, fname='images.jpg', names=None):

    """
    Plot image grid with labels and pred
    :param im: numpy array of image
    :param target: (n,6) ((x1,y1,x2,y2,cls, is_matched),...)
    :param pred: (k,7) ((x1,y1,x2,y2,score,cls, is_matched),...)
    :param paths: one image path
    :param fname: visual image names
    :param names: list,  class names
    :return:
    """
    if isinstance(target,tuple):
        targets,seg_tar = target  # detect_out  seg_out(bs,h,w)
        seg_mask = torch.nn.functional.one_hot(seg_tar)
        if isinstance(seg_mask,torch.Tensor):
            seg_mask =seg_mask.cpu().float().numpy()*255
    else:
        seg_mask=None
    if isinstance(im, torch.Tensor):
        im = im.cpu().float().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    if np.max(im[0]) <= 1:
        im *= 255  # de-normalise (optional)
    h, w, _ = im.shape  # height, width, channal

    # Build Image
    im_t_p = np.full((int(h), int(2 * w), 3), 255, dtype=np.uint8)  # init

    # add seg mask to image
    if seg_mask is not None:
        im = seg_mask * 0.3 + im * 0.7
        im = im.astype(int)
        im = np.where(im > 255, 255, im).astype(np.uint8)
    im_t_p[0: h, 0: w, :] = im
    im_t_p[0: h, w: 2 * w, :] = im
    # Annotate
    fs = int((h + w) * 0.01)  # font size
    annotator = Annotator(im_t_p, line_width=round(fs / 10), font_size=fs, pil=True)

    flag = [ ' GT', ' Pred']
    for i, box_c in enumerate([target, pred]):
        x, y = i*int(w), 0  # block origin
        annotator.rectangle([x, y, x + w, y + h], None, (255, 255, 255), width=2)  # borders
        if paths:
            annotator.text((x + 5, y - 5 + h), text=Path(paths).name[:40].replace(":", "_")+flag[i], txt_color=(220, 220, 220))  # filenames

        # boxes = xywh2xyxy(box_c[:, :4]).T


        labels = box_c.shape[1] == 6  # labels if no conf column
        conf = None if labels else box_c[:, 4]  # check for confidence presence (label vs pred)

        classes = box_c[:, -2].astype('int')
        boxes = box_c[:, :4].T
        matched = box_c[:, -1].astype('int')

        if boxes.shape[1]:
            if boxes.max() <= 1.01:  # if normalized with tolerance 0.01
                boxes[[0, 2]] *= w  # scale to pixels
                boxes[[1, 3]] *= h
        boxes[[0, 2]] += x
        boxes[[1, 3]] += y
        for j, box in enumerate(boxes.astype(np.int).T.tolist()):
            cls = classes[j] # category from 1
            color = colors(cls)
            cls = names[cls] if names else cls
            if not matched[j]:
                annotator.mark(box)

            if seg_mask is None:
                label = f'{cls}' if labels else f'{cls} {conf[j]:.1f}'
            else:
                label = ''
            annotator.box_label(box, label, color=color)
    annotator.im.save(fname)  # save

def plot_lr_scheduler(optimizer, scheduler, epochs=300, save_dir=''):
    # Plot LR simulating training for full epochs
    optimizer, scheduler = copy(optimizer), copy(scheduler)  # do not modify originals
    y = []
    for _ in range(epochs):
        scheduler.step()
        y.append(optimizer.param_groups[0]['lr'])
    plt.plot(y, '.-', label='LR')
    plt.xlabel('epoch')
    plt.ylabel('LR')
    plt.grid()
    plt.xlim(0, epochs)
    plt.ylim(0)
    plt.savefig(Path(save_dir) / 'LR.png', dpi=200)
    plt.close()

def hist2d(x, y, n=100):
    # 2d histogram used in labels.png and evolve.png
    xedges, yedges = np.linspace(x.min(), x.max(), n), np.linspace(y.min(), y.max(), n)
    hist, xedges, yedges = np.histogram2d(x, y, (xedges, yedges))
    xidx = np.clip(np.digitize(x, xedges) - 1, 0, hist.shape[0] - 1)
    yidx = np.clip(np.digitize(y, yedges) - 1, 0, hist.shape[1] - 1)
    return np.log(hist[xidx, yidx])

def plot_val_txt():  # from utils.plots import *; plot_val()
    # Plot val.txt histograms
    x = np.loadtxt('val.txt', dtype=np.float32)
    box = xyxy2xywh(x[:, :4])
    cx, cy = box[:, 0], box[:, 1]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
    ax.hist2d(cx, cy, bins=600, cmax=10, cmin=0)
    ax.set_aspect('equal')
    plt.savefig('hist2d.png', dpi=300)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
    ax[0].hist(cx, bins=600)
    ax[1].hist(cy, bins=600)
    plt.savefig('hist1d.png', dpi=200)

def plot_targets_txt():  # from utils.plots import *; plot_targets_txt()
    # Plot targets.txt histograms
    x = np.loadtxt('targets.txt', dtype=np.float32).T
    s = ['x targets', 'y targets', 'width targets', 'height targets']
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
    ax = ax.ravel()
    for i in range(4):
        ax[i].hist(x[i], bins=100, label=f'{x[i].mean():.3g} +/- {x[i].std():.3g}')
        ax[i].legend()
        ax[i].set_title(s[i])
    plt.savefig('targets.jpg', dpi=200)

# def plot_results(file='path/to/results.csv', dir=''):
#     # Plot training results.csv. Usage: from utils.plots import *; plot_results('path/to/results.csv')
#     save_dir = Path(file).parent if file else Path(dir)
#     fig, ax = plt.subplots(2, 5, figsize=(12, 6), tight_layout=True)
#     ax = ax.ravel()
#     files = list(save_dir.glob('results*.csv'))
#     assert len(files), f'No results.csv files found in {save_dir.resolve()}, nothing to plot.'
#     for fi, f in enumerate(files):
#         try:
#             data = pd.read_csv(f)
#             s = [x.strip() for x in data.columns]
#             x = data.values[:, 0]
#             for i, j in enumerate([1, 2, 3, 4, 5, 8, 9, 10, 6, 7]):
#                 y = data.values[:, j]
#                 # y[y == 0] = np.nan  # don't show zero values
#                 ax[i].plot(x, y, marker='.', label=f.stem, linewidth=2, markersize=8)  #f.stem = 'results'
#                 ax[i].set_title(s[j], fontsize=12)
#                 # if j in [8, 9, 10]:  # share train and val loss y axes
#                 #     ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])
#         except Exception as e:
#             print(f'Warning: Plotting error for {f}: {e}')
#     ax[1].legend()
#     fig.savefig(save_dir / 'results.png', dpi=200)
#     plt.close()

def make_divide(n, div):
    dived =  n // div
    if n > dived*div:
       dived+=1
    return dived
def plot_results(file='path/to/results.csv', save_name='results.png',dir=''):
    # Plot training results.csv. Usage: from utils.plots import *; plot_results('path/to/results.csv')
    save_dir = Path(file).parent if file else Path(dir)
    files = list(save_dir.glob('results*.csv'))
    assert len(files), f'No results.csv files found in {save_dir.resolve()}, nothing to plot.'

    for fi, f in enumerate(files):
        try:
            data = pd.read_csv(f)
            s = [x.strip()  if 'lr2' not in x else 'bais_l_r' for x in data.columns]
            x = data.values[:, 0]
            if 'seg' not in save_name:
                new_title = set([title.split('/')[-1]  if 'lr' not in title else 'lr' for title in s[1:]])
                fig, ax = plt.subplots(3, make_divide(len(new_title), 3), figsize=(24, 16), tight_layout=True)
            else:
                new_title = set([title.split('/')[-1] for title in s[1:] if 'seg' in title].sort())
                fig, ax = plt.subplots(2, make_divide(len(new_title), 2), figsize=(12, 8), tight_layout=True)

            ax = ax.ravel()
            for i,now_title in enumerate(new_title):
                for j in range(1, len(s)):
                    s_p = s[j].split('/')
                    if now_title=='lr':
                        if now_title in s_p[-1]:
                            y = data.values[:, j]
                            label = s_p[-1]
                            ax[i].plot(x, y, marker='.', label=label, linewidth=2, markersize=8)  # f.stem = 'results'
                            ax[i].set_title(now_title, fontsize=12)
                            ax[i].legend(loc=1)
                    if now_title == s_p[-1]:
                        y = data.values[:, j]
                        # y[y == 0] = np.nan  # don't show zero values
                        if 'loss' in now_title:
                            label = s_p[0]
                        else:
                            label = ''
                        ax[i].plot(x, y, marker='.', label=label, linewidth=3, markersize=8)  #f.stem = 'results'
                        ax[i].set_title(now_title, fontsize=12)
                        # if j in [8, 9, 10]:  # share train and val loss y axes
                        #     ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])
                        if label:
                            ax[i].legend(loc=1)
            # plt.legend(loc=2)
            fig.savefig(save_dir / save_name, dpi=200)
            plt.close()
            print(f"fig saved in {save_dir}/{save_name}")
        except Exception as e:
            print(f'Warning: Plotting error for {f}: {e}')

def plot_evolve(evolve_csv='path/to/evolve.csv'):  # from utils.plots import *; plot_evolve()
    # Plot evolve.csv hyp evolution results
    evolve_csv = Path(evolve_csv)
    data = pd.read_csv(evolve_csv)
    keys = [x.strip() for x in data.columns]
    x = data.values
    from utils.calculate import fitness
    f = fitness(x)
    j = np.argmax(f)  # max fitness index
    plt.figure(figsize=(10, 12), tight_layout=True)
    matplotlib.rc('font', **{'size': 8})
    for i, k in enumerate(keys[7:]):
        v = x[:, 7 + i]
        mu = v[j]  # best single result
        plt.subplot(6, 5, i + 1)
        plt.scatter(v, f, c=hist2d(v, f, 20), cmap='viridis', alpha=.8, edgecolors='none')
        plt.plot(mu, f.max(), 'k+', markersize=15)
        plt.title(f'{k} = {mu:.3g}', fontdict={'size': 9})  # limit to 40 characters
        if i % 5 != 0:
            plt.yticks([])
        print(f'{k:>15}: {mu:.3g}')
    f = evolve_csv.with_suffix('.png')  # filename
    plt.savefig(f, dpi=200)
    plt.close()
    print(f'Saved {f}')

def plot_labels(labels, names=(), save_dir=Path(''), logger=LOGGER):
    # plot dataset labels
    logger.info(f"Plotting labels to {save_dir / 'labels.jpg'}... ")
    c, b = labels[:, 0], labels[:, 1:].transpose()  # classes, boxes
    nc = int(c.max() + 1)  # number of classes
    x = pd.DataFrame(b.transpose(), columns=['x', 'y', 'width', 'height'])

    # seaborn correlogram
    sn.pairplot(x, corner=True, diag_kind='auto', kind='hist', diag_kws=dict(bins=50), plot_kws=dict(pmax=0.9))
    plt.savefig(save_dir / 'labels_correlogram.jpg', dpi=200)
    plt.close()

    # matplotlib labels
    matplotlib.use('svg')  # faster
    ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)[1].ravel()
    y = ax[0].hist(c, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    # [y[2].patches[i].set_color([x / 255 for x in colors(i)]) for i in range(nc)]  # update colors bug #3195
    ax[0].set_ylabel('instances')
    if 0 < len(names) < 30:
        ax[0].set_xticks(range(len(names)))
        ax[0].set_xticklabels(names, rotation=90, fontsize=10)
    else:
        ax[0].set_xlabel('classes')
    sn.histplot(x, x='x', y='y', ax=ax[2], bins=50, pmax=0.9)
    sn.histplot(x, x='width', y='height', ax=ax[3], bins=50, pmax=0.9)

    # rectangles
    labels[:, 1:3] = 0.5  # center
    labels[:, 1:] = xywh2xyxy(labels[:, 1:]) * 2000
    img = Image.fromarray(np.ones((2000, 2000, 3), dtype=np.uint8) * 255)
    for cls, *box in labels[:1000]:
        ImageDraw.Draw(img).rectangle(box, width=1, outline=colors(cls))  # plot
    ax[1].imshow(img)
    ax[1].axis('off')

    for a in [0, 1, 2, 3]:
        for s in ['top', 'right', 'left', 'bottom']:
            ax[a].spines[s].set_visible(False)

    plt.savefig(save_dir / 'labels.jpg', dpi=200)
    matplotlib.use('Agg')
    plt.close()

def save_one_box(xyxy, im, file='image.jpg', gain=1.02, pad=10, square=False, BGR=False, save=True):
    # Save image crop as {file} with crop size multiple {gain} and {pad} pixels. Save and/or return crop
    xyxy = torch.tensor(xyxy).view(-1, 4)
    b = xyxy2xywh(xyxy)  # boxes
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square
    b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
    xyxy = xywh2xyxy(b).long()
    clip_coords(xyxy, im.shape)
    crop = im[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::(1 if BGR else -1)]
    if save:
        file.parent.mkdir(parents=True, exist_ok=True)  # make directory
        cv2.imwrite(str(increment_path(file).with_suffix('.jpg')), crop)
    return crop


def output_to_target(output):
    # Convert model output to target format [batch_id, class_id, x, y, w, h, conf]
    targets = []
    for i, o in enumerate(output):
        for *box, conf, cls in o.cpu().numpy():
            targets.append([i, cls, *list(*xyxy2xywh(np.array(box)[None])), conf])
    return np.array(targets)


def visual_match_pred(match_id, pred, target, img_path, save_dir='.', names=(),conf_t=0.1):
    """
    visualize the pred box and gt box via the rectangle matched
    :param match_id:array (n,2), n is gtbox number, ((gt_id, pred_id),...)
    :param pred: array(k,6), k is predbox number, ((x1,y1,x2,y2,score,cls),...)
    :param target: array(n,5), n is gtbox number, ((x1,y1,x2,y2,cls),...)
    :param img_path:  one image path
    :param save_dir: saved dir of the result, and under the dir, ./visual_images/matched which saved the matched visual images
    :param names: list,  class names
    :param conf_t: conf thresh
    :return:
    """
    img_name = os.path.split(img_path)[-1]
    # if not isinstance(img_path, str):
    #     img_path = str(img_path)
    # Sort by objectness
    p_ids = np.where(pred[:, 4]>=conf_t)[0].tolist() #(array([],)[0] to list

    gt_result = np.zeros((target.shape[0],target.shape[1]+1))
    gt_result[:, :-1] = target

    bbox_result = np.zeros((len(p_ids), pred.shape[1]+1))
    bbox_result[:, :-1] = pred[p_ids]

    conf_filter_matched = []
    for pe, p_id in enumerate(match_id[:, 1]):
        if p_id in p_ids:
             conf_filter_matched.append(pe)

    conf_filter_matched_id = match_id[conf_filter_matched]
    gm = conf_filter_matched_id.shape[0]
    gt_result[conf_filter_matched_id[:, 0], -1] = 1
    bbox_result[conf_filter_matched_id[:, 1], -1] = 1

    if gm!=gt_result.shape[0]:
        is_igmatch = 'not_matched'
    elif gm!=bbox_result.shape[0]:
        is_igmatch = 'false_pred'
    else:
        is_igmatch = 'matched'

    save_path = os.path.join(save_dir,  'visual_images', is_igmatch)
    mkdir(save_path)

    outfile = os.path.join(save_path, img_name)
    im = cv2.imread(str(img_path))  # BGR
    Thread(target=visual_images, args=(im, gt_result, bbox_result, img_path, outfile, names),
           daemon=True).start()

def save_object(ims, targets, preds, paths=None, save_dir='exp', visual_task=0):

    """
    Plot image grid with labels and pred
    :param im: numpy array of image
    :param target: cls label  (bs, 1)
    :param pred: (bs,1)
    :param paths: image paths
    :return:
    """
    save_dir  = os.path.join(save_dir, 'filter_false')
    mkdir(save_dir)
    if isinstance(ims, torch.Tensor):
        ims = ims.cpu().permute(0, 2, 3, 1).float().numpy()  # bs, _, h, w to bs, h,w, _
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if np.max(ims[0]) <= 1:
        ims *= 255  # de-normalise (optional)
    i_l = np.arange(0, len(ims)).reshape(-1,1)[preds!=targets]
    visual_idx = np.logical_or(preds[i_l]==1, targets[i_l]==1) if visual_task==2 else (preds if visual_task else targets)[i_l]==1
    i_l = i_l.reshape(-1,1)[visual_idx].flatten().tolist()
    pred_flag = [ 'pred_bg' , 'pred_object']
    for i in i_l:
        im = ims[i].astype(np.uint8)
        target = targets[i]
        pred = preds[i]
        path = paths[i]
        h, w, _ = im.shape  # height, width, channal
        # img = np.full((int(h), int(w), 1), 255, dtype=np.uint8)  # init
        # Annotate
        fs = int((h + w) * 0.01)  # font size
        annotator = Annotator(im.repeat(3, axis=2), line_width=round(fs / 10), font_size=fs, pil=True)

        for i, box_c in enumerate([target, pred]):
            x, y = i*int(w), 0  # block origin
            annotator.rectangle([x, y, x + w, y + h], None, (255, 255, 255), width=2)  # borders
            if path:
                annotator.text((x + 5, y - 5 + h), text=Path(path).name[:40].replace(":", "_")+pred_flag[int(pred)], txt_color=(220, 220, 220))  # filenames
        annotator.im.save(os.path.join(save_dir, Path(path).name))  # save