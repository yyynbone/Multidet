"""
Auto-anchor utils
"""

import random
import numpy as np
import torch
import yaml
from tqdm import tqdm
from utils import print_log, colorstr, xyxy2xywh
from dataloader import read_image
import matplotlib.pyplot as plt
PREFIX = colorstr('AutoAnchor: ')

def check_anchor_order(m, loggger=None):
    # Check anchor order against stride order for YOLOv5 Detect() module m, and correct if necessary
    a = m.anchors.prod(-1).view(-1)  # anchor area
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da.sign() != ds.sign():  # same order
        loggger.info(f'{PREFIX}Reversing anchor order')
        m.anchors[:] = m.anchors.flip(0)


def run_anchor(dataset, model, thr=4.0, imgsz=640, logger=None):
    det = model.module[model.module.det_head_idx] if hasattr(model, 'module') else model[model.det_head_idx]
    anchor_num = det.na * det.nl
    new_anchors = kmean_anchors(dataset, n=anchor_num, img_size=imgsz, thr=thr, gen=1000, verbose=False, logger=logger)
    # new_anchors = np.array([5, 6, 9, 7, 8, 13, 16, 12, 12, 28, 24, 19, 43, 33, 80, 64, 166, 133])
    new_anchors = torch.tensor(new_anchors, device=det.anchors.device).type_as(det.anchors)
    det.anchors[:] = new_anchors.clone().view_as(det.anchors) / det.stride.to(det.anchors.device).view(-1, 1, 1)  # loss
    check_anchor_order(det, logger)
    print_log(str(det.anchors), logger)
    print_log('New anchors saved to model. Update model config to use these anchors in the future.', logger)

def check_anchors(dataset, model, imgsz=640, thr=4.0, logger=None):
    # Check anchor fit to data, recompute if necessary
    m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1] # Detect()
    if hasattr(m, 'anchors'):
        if len(m.anchors):
            resized_wh = []
            for result in dataset.results:
                read_image(result)
                ratio_w = result.get('rect_shape', result['img_size'])[1]/ result['ori_shape'][1]
                ratio_h = result.get('rect_shape', result['img_size'])[0] / result['ori_shape'][0]
                wh = (result['labels'][:, 3:5] - result['labels'][:, 1:3] )* np.array([ratio_w, ratio_h])
                resized_wh.append(wh)

            scale = np.random.uniform(0.95, 1.05, size=(len(resized_wh), 1)) # augment scale
            wh = torch.tensor(np.concatenate([l * s for s, l in zip(scale, resized_wh)])).float()  # wh

            def metric(k):  # compute metric
                # print(wh.shape, k)
                r = wh[:, None] / k[None]
                x = torch.min(r, 1 / r).min(2)[0]  # ratio metric
                best = x.max(1)[0]  # best_x
                aat = (x > 1 / thr).float().sum(1).mean()  # anchors above threshold
                bpr = (best > 1 / thr).float().mean()  # best possible recall
                return bpr, aat

            anchors = m.anchors.clone() * m.stride.to(m.anchors.device).view(-1, 1, 1)  # current anchors
            anchors  = anchors.cpu().view(-1, 2)
            bpr, aat = metric(anchors)
            s = f'\n{PREFIX}{aat:.2f} anchors/target, {bpr:.3f} Best Possible Recall (BPR). '
            if bpr > 0.98:  # threshold to recompute
                print_log(f'{s}Current anchors are a good fit to dataset', logger)
                print_log(anchors, logger)

            else:
                print_log(f'{s}Anchors are a poor fit to dataset, attempting to improve...', logger)
                na = m.anchors.numel() // 2  # number of anchors
                # try:
                #     anchors = kmean_anchors(dataset, n=na, thr=thr, gen=1000, verbose=False, logger=logger)
                # except Exception as e:
                #     print_log(f'{PREFIX}ERROR: {e}', logger)
                anchors = kmean_anchors(dataset, n=na, img_size=imgsz, thr=thr, gen=1000, verbose=False, logger=logger)   #return numpy

                new_bpr = metric(anchors)[0]
                if new_bpr > bpr:  # replace anchors
                    anchors = torch.tensor(anchors, device=m.anchors.device).type_as(m.anchors)
                    m.anchors[:] = anchors.clone().view_as(m.anchors) / m.stride.to(m.anchors.device).view(-1, 1, 1)  # loss
                    check_anchor_order(m, logger)
                    print_log(f'{PREFIX}New anchors saved to model. Update model *.yaml to use these anchors in the future.', logger)
                else:
                    print_log(f'{PREFIX}Original anchors better than new anchors. Proceeding with original anchors.', logger)

def kmean_anchors(dataset='./data/coco128.yaml', n=9, img_size=640, thr=4.0, gen=1000, verbose=True, logger=None):
    """ Creates kmeans-evolved anchors from training dataset

        Arguments:
            dataset: path to data.yaml, or a loaded dataset
            n: number of anchors
            img_size: image size used for training
            thr: anchor-label wh ratio threshold hyperparameter hyp['anchor_t'] used for training, default=4.0
            gen: generations to evolve anchors using genetic algorithm
            verbose: print all results

        Return:
            k: kmeans evolved anchors , numpy

        Usage:
            from utils.autoanchor import *; _ = kmean_anchors()
    """
    from scipy.cluster.vq import kmeans

    thr = 1 / thr

    def metric(k, wh):  # compute metrics
        r = wh[:, None] / k[None]
        x = torch.min(r, 1 / r).min(2)[0]  # ratio metric
        # x = wh_iou(wh, torch.tensor(k))  # iou metric
        return x, x.max(1)[0]  # x, best_x

    def anchor_fitness(k):  # mutation fitness
        _, best = metric(torch.tensor(k, dtype=torch.float32), wh)
        return (best * (best > thr).float()).mean()  # fitness

    def print_results(k, verbose=True):
        k = k[np.argsort(k.prod(1))]  # sort small to large
        x, best = metric(k, wh0)
        bpr, aat = (best > thr).float().mean(), (x > thr).float().mean() * n  # best possible recall, anch > thr
        s = f'{PREFIX}thr={thr:.2f}: {bpr:.4f} best possible recall, {aat:.2f} anchors past thr\n' \
            f'{PREFIX}n={n}, metric_all={x.mean():.3f}/{best.mean():.3f}-mean/best, ' \
            f'past_thr={x[x > thr].mean():.3f}-mean: '
        for i, x in enumerate(k):
            s += '%i,%i, ' % (round(x[0]), round(x[1]))
        if verbose:
            print_log(s[:-2], logger)
        return k

    if isinstance(dataset, str):  # *.yaml file
        from dataloader import get_dataset
        from utils import select_class_tuple, check_yaml, check_dataset

        with open(check_yaml(dataset), errors='ignore') as f:
            data_dict = yaml.safe_load(f)  # model dict
        data_dict = check_dataset(data_dict)
        with open(check_yaml('data_preprocess.yaml'), errors='ignore') as f:
            data_pre = yaml.safe_load(f)  # load hyps dict

        results, _ = get_dataset(data_dict['train'],
                              data_pre['val'],
                              img_size=img_size,
                              logger=logger,
                              select_class=select_class_tuple(data_dict))
        img_size = None
    else:
        results = dataset.results
    resized_wh = []
    for result in results:
        if img_size is not None:
            read_image(result)
            ratio_w = result.get('rect_shape', result['img_size'])[1] / result['ori_shape'][1]
            ratio_h = result.get('rect_shape', result['img_size'])[0] / result['ori_shape'][0]
        else:
            ratio_w, ratio_h = 1, 1
        wh = (result['labels'][:, 3:5] - result['labels'][:, 1:3] ) * np.array([ratio_w, ratio_h])
        resized_wh.append(wh)
    wh0 = torch.tensor(np.concatenate(resized_wh)).float().cpu() # wh

    # Filter
    i = (wh0 < 3.0).any(1).sum()     #any(dim)
    if i:
        print_log(f'{PREFIX}WARNING: Extremely small objects found. {i} of {len(wh0)} labels are < 3 pixels in size.', logger)
    wh = wh0[(wh0 >= 2.0).any(1)]  # filter > 2 pixels
    # wh = wh * (np.random.rand(wh.shape[0], 1) * 0.9 + 0.1)  # multiply by random scale 0-1

    # Kmeans calculation
    print_log(f'{PREFIX}Running kmeans for {n} anchors on {len(wh)} points...', logger)
    s = wh.std(0)  # sigmas for whitening
    k, dist = kmeans(wh / s, n, iter=30)  # points, mean distance
    assert len(k) == n, f'{PREFIX}ERROR: scipy.cluster.vq.kmeans requested {n} points but returned only {len(k)}'
    k = k * s.numpy()
    # wh = torch.tensor(wh, dtype=torch.float32)  # filtered
    # wh0 = torch.tensor(wh0, dtype=torch.float32)  # unfiltered
    k = print_results(k, verbose=False)
    # print('kmeans down')
    # Plot
    # print('now plot')
    # k, d = [None] * 20, [None] * 20
    # for i in tqdm(range(1, 21)):
    #     k[i-1], d[i-1] = kmeans(wh / s, i)  # points, mean distance
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7), tight_layout=True)
    # ax = ax.ravel()
    # ax[0].plot(np.arange(1, 21), np.array(d) ** 2, marker='.')
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7))  # plot wh
    # ax[0].hist(wh[wh[:, 0]<100, 0],400)
    # ax[1].hist(wh[wh[:, 1]<100, 1],400)
    # fig.savefig('wh.png', dpi=200)
    # print('plot down')
    # Evolve
    npr = np.random
    f, sh, mp, s = anchor_fitness(k), k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
    pbar = tqdm(range(gen), desc=f'{PREFIX}Evolving anchors with Genetic Algorithm:')  # progress bar
    for _ in pbar:
        v = np.ones(sh)
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
            v = ((npr.random(sh) < mp) * random.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg = anchor_fitness(kg)
        if fg > f:
            f, k = fg, kg.copy()
            pbar.desc = f'{PREFIX}Evolving anchors with Genetic Algorithm: fitness = {f:.4f}'
            if verbose:
                print_results(k, verbose)

    return print_results(k)

if __name__ == '__main__':
    dataset = 'drone.yaml'
    anchors = kmean_anchors(dataset, n=9, img_size=640, thr=4, gen=1000, verbose=False)  # return numpy
    print(anchors)