import os
import argparse
import json
import random
from pathlib import Path
from threading import Thread
import numpy as np
import torch
from tqdm import tqdm
from glob import glob
from torch.utils.data import DataLoader

from utils import ROOT
from models import attempt_load
from loss import *
from dataloader import LoadImagesAndLabels


from utils import ( set_logging, check_dataset, check_img_size, check_yaml, ConfusionMatrix, calcu_per_class,
                    output_to_target, xywh2xyxy, xyxy2xywh, save_one_txt,save_one_json, increment_path,
                    scale_coords, non_max_suppression_with_iof, plot_images, select_device, time_sync, print_args,
                    div_area_idx, process_batch, area_filter, visual_match_pred, print_log, visual_return,
                    select_score_from_json, select_class_tuple, classify_match, save_object)
FILE = Path(__file__).resolve()
# LOGGER = set_logging(__name__)

@torch.no_grad()
def run(data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,  # save a COCO-JSON results file
        half=True,  # use FP16 half-precision inference
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        compute_loss=None,
        bgr=1,
        div_area=None,
        last_conf=0.1,
        loss_num=3,
        class_map=list(range(1, 81)), # class_map save to json, so categoryid from 1.
        visual_matched=False,
        logger=None,
        task='val',
        **kwargs
        ):
    # if visual_matched:
    #     save_json = True
    if not isinstance(last_conf, (list, tuple)):
        last_conf = [last_conf]
    if bgr:
        ch_in = 3
    else:
        ch_in = 1
    if not device:
        device = next(model.parameters()).device
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    model.half() if half else model.float()

    if isinstance(imgsz, int):
        imgsz = (imgsz, imgsz)
    elif len(imgsz)==1:
        imgsz = (imgsz[0], imgsz[0])
    else:
        imgsz = tuple(imgsz)
    # Configure
    model.eval()
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith('val.json')  # COCO dataset
    # nc = len(model['names'])  if hasattr(model, 'names') else int(data['nc'])  # number of classes
    nc = model.nc if hasattr(model, 'nc') else int(data['nc']) # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)

    names =  {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else [])}
    for ci, cls in enumerate(data.get('names', [])):
        names[ci] = cls

    s = ('%20s' + '%11s' * 7) % ('Class', 'Labels', 'R_num', 'P_num', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    loss = torch.zeros(loss_num, device=device)  # mean losses
    jdict, ap, ap_class = [], [], []
    stats, cls_stats, cls_stats_indet = [], [], []
    div_class_num = 1
    if isinstance(div_area, (list, tuple)):
        div_class_num += len(div_area) + 1
    dataloader.dataset.mosaic = False
    pbar = tqdm(dataloader, desc='validate detection', bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar

    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):

        # if batch_i >1:
        #     break
        seen += len(im)
        t1 = time_sync()
        im = im.to(device, non_blocking=True)
        targets = [t.to(device) if t is not None else t for t in targets]


        im = im.half() if half else im.float()  # uint8 to fp16/32
        # print(im.shape)
        im /= 255  # 0 - 255 to 0.0 - 1.0

        # warmup
        if weights is not None and batch_i==0:
            if isinstance(model.device, torch.device) and model.device.type != 'cpu':  # only warmup GPU models
                warm_im = torch.zeros_like(im[:1], device=device)
                model.forward(warm_im)  # warmup

        nb, _, height, width = im.shape  # batch size, channels, height, width
        t2 = time_sync()
        dt[0] += t2 - t1
        # Inference
        head_out = model(im, augment=augment)
        # if model.out_det_from:
        #     filter_bool = head_out[0].cpu().numpy()  # classify out image index bool
        #     im = im[filter_bool]
        #     paths = paths[filter_bool]
        #     shapes = shapes[filter_bool]
        # if model.filter_bool is not None:
        #     logger.info("now we select object image use filter")
        out, train_out =  head_out[model.out_det_from]  # detect out

        dt[1] += time_sync() - t2
        # Loss
        if compute_loss:
            loss_batch, loss_items= compute_loss(train_out, targets)  # box, obj, cls, seg
            # print(loss_batch, loss_items)
            loss += loss_items

        class_label, targets, seg_img = targets
        if loss_num > 1:
            # NMS
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            t3 = time_sync()
            out = non_max_suppression_with_iof(out, conf_thres, iou_thres, labels=lb, multi_label=True, iof_nms=True)
            dt[2] += time_sync() - t3

            # Metrics
            for si, pred in enumerate(out):
                labels = targets[targets[:, 0] == si, 1:]
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []  # target class
                path, shape = Path(paths[si]), shapes[si][0]


                if len(pred) == 0:
                    if nl:
                        area_idx = torch.zeros(nl, div_class_num, dtype=torch.bool)
                        area = (labels[:, 2] * labels[:, 3]).cpu()
                        area_idx = div_area_idx(area, div_area, area_idx)
                        stats.append((torch.zeros(0, div_class_num, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls, area_idx))
                    continue

                # Predictions
                predn = pred.clone()   # predn is a pred from scaled image to origin size image
                scale_coords(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

                # Evaluate
                if nl:
                    tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                    scale_coords(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                    labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                    # add target idx of different gt box
                    correct, area_idx, matches = process_batch(predn, labelsn, iouv, div_area=div_area)  #iou 0.5-0.95 共10个
                    if plots:
                        confusion_matrix.process_batch(predn, labelsn)
                else:
                    correct = torch.zeros(predn.shape[0], div_class_num, niou, dtype=torch.bool)
                    # attention: now we give the all area level of pred to False,
                    # if it's all, in different area level, pred_num is less then all pred in fact
                    # aimed to assign the pred to different level， divided by the pred area
                    if div_class_num>1:
                        pred_box = xyxy2xywh(predn[:, :4])
                        area = (pred_box[:, 2] * pred_box[:, 3]).cpu()
                        correct = div_area_idx(area, div_area, correct)

                    area_idx = torch.zeros(0, div_class_num, dtype=torch.bool)

                stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls, area_idx))  # (correct, conf, pcls, tcls)

                # Save/log
                if save_txt:
                    save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / (path.stem + '.txt'))
                if save_json:
                    save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary

                if visual_matched:
                    pred_b_s_l = pred[:, :6].cpu().numpy()  # （300，6）
                    box_l = np.zeros((0, 5))
                    match_id = np.zeros((0, 2))
                    if nl:
                        box_l = torch.cat((xywh2xyxy(labels[:, 1:5]), labels[:, 0:1]), 1).cpu().numpy()  # (29,5) float
                        match_id = matches[:, :2].cpu().numpy()  # [label, detection], (29,2)
                    # match_id is float
                    match_id = match_id.astype(int)
                    # # if match_id.shape[0] != box_l.shape[0]:
                    # #     print("gt not matched")
                    visual_match_pred(im[si], match_id, pred_b_s_l, box_l, path, save_dir=save_dir, names=names,
                                      conf_ts=last_conf) #correct is shape of [n, div_area=4, iou=10]
            # Plot images
            if plots and batch_i % (1280//batch_size) == 5:
            # if plots and random.randint(1, 100) < 20 if len(pbar)>10 else 100:
                plot_tar = targets
                plot_pred = output_to_target(out)
                f = save_dir / f'val_batch{batch_i}_labels.jpg'  # labels
                Thread(target=plot_images, args=(im, plot_tar, paths, f, names), daemon=True).start()
                f = save_dir / f'val_batch{batch_i}_pred.jpg'  # predictions
                Thread(target=plot_images, args=(im, plot_pred, paths, f, names), daemon=True).start()
                # plot_images(im, plot_pred, paths, f, names)
        elif loss_num == 1:
            cls_stats.append((out.cpu(), class_label.cpu()))
            if visual_matched:
                save_object(im ,class_label.cpu(), out.cpu(), paths, save_dir=save_dir, conf_ts=last_conf, f_map=train_out)


        if model.out_det_from == 1:
            out, _ = head_out[0]  # detect out
            cls_stats_indet.append((out.cpu(), class_label.cpu()))
            if visual_matched:
                save_object(im ,class_label.cpu(), out.cpu(), paths, save_dir=save_dir, conf_ts=(getattr(model, 'class_conf', 0.4)))

    # Compute metrics
    ps, rs, accus = [], [], []

    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    cls_stats = [np.concatenate(x, 0) for x in zip(*cls_stats)]  # to numpy
    cls_stats_indet = [np.concatenate(x, 0) for x in zip(*cls_stats_indet)]  # to numpy

    if len(cls_stats):
        ps, rs , accus = classify_match(*cls_stats, conf_ts=last_conf, logger_func=logger.info)
        dt[2] += 0.
        det_result = (ps[1], rs[1], 0, accus[0], (loss.cpu() / len(dataloader)).tolist())
        # Print speeds
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        shape = (batch_size, ch_in, imgsz[0], imgsz[1])
        logger.info(
            f'all image number is {seen}, which cost {sum(dt)}s\nSpeed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)
        return det_result, 0, t

    if len(cls_stats_indet):
        _, _, _ = classify_match(*cls_stats_indet, conf_ts=(getattr(model, 'class_conf', 0.4)), logger_func=logger.info)

    if len(stats) and stats[0].any():

        ps, rs, pred_truths, lev_nps, ap, ap_class,  nt = calcu_per_class(*stats, plot=plots, save_dir=save_dir, names=names, conf_ts=last_conf)
        ap50, ap = ap[..., 0], ap.mean(2)  # AP@0.5, AP@0. 5:0.95
        for li, conf_t in enumerate(last_conf):
            p, r, pred_truth, lev_np = ps[li], rs[li], pred_truths[li], lev_nps[li]
            # Print results
            logger.info(s)
            pf = '%20s' + '%11i' * 3 + '%11.3g' * 4  # print format
            all_tp, all_p, all_t = pred_truth[:, 0].sum(), lev_np[:, 0].sum(), nt[:, 0].sum()

            mp, mr = all_tp / max(all_p, 1e-16), all_tp / max(all_t, 1e-16)
            map50, map = ap50[:, 0].mean(), ap[:, 0].mean()
            logger.info(pf % (f'all_conf_{conf_t}', all_t, all_tp, all_p, mp, mr, map50, map))
            if div_area is not None:
                f_t, f_tp, f_p = area_filter(nt), area_filter(pred_truth), area_filter(lev_np)
                f_map50, f_map = area_filter(ap50), area_filter(ap)
                f_map50 = (f_map50 * f_p).sum()/ max(f_p.sum(), 1e-16)
                f_map = (f_map * f_p).sum() / max(f_p.sum(), 1e-16)
                f_t, f_tp, f_p = f_t.sum(), f_tp.sum(), f_p.sum()
                f_mp, f_mr = f_tp / max(f_p, 1e-16), f_tp / max(f_t, 1e-16)
                logger.info(pf % (f'all_over_{div_area[0]}', f_t, f_tp, f_p, f_mp, f_mr, f_map50, f_map))

            # Print results per class
            if verbose and nc > 1 and len(stats):
                for i, c in enumerate(ap_class):
                    if c in names:
                        logger.info(pf % (names[c], nt[i, 0], pred_truth[i, 0], lev_np[i, 0], p[i, 0], r[i, 0], ap50[i, 0], ap[i, 0]))
                        if isinstance(div_area, (list, tuple)):
                            for div_i, div_ang in enumerate(div_area):
                                logger.info(pf % (f'{names[c]}_under_{div_ang}', nt[i, div_i+1], pred_truth[i, div_i+1], lev_np[i, div_i+1], p[i, div_i+1], r[i, div_i+1], ap50[i, div_i+1], ap[i, div_i+1]))
                            logger.info(pf % (
                            f'{names[c]}_over_{div_ang}', nt[i, -1], pred_truth[i, -1], lev_np[i, -1],
                            p[i, -1], r[i, -1], ap50[i, -1], ap[i, -1]))

    else:
        logger.info('none labels')

    # Print speeds
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image

    shape = (batch_size, ch_in, imgsz[0], imgsz[1])
    logger.info(f'all image number is {seen}, which cost {sum(dt)}s\nSpeed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))


    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        # anno_json = str(Path(data.get('path', '../coco')) / 'annotations/instances_val2017.json')  # annotations json
        anno_json = str(Path(data.get('path', '../coco')) / f'COCO/annotation/{task}.json')  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        logger.info(f'\nEvaluating pycocotools mAP... saving {pred_json}...')

        ######### for Imageid  of filename 2 Imageid ################
        with open(anno_json, 'r') as f:
            images = json.load(f)['images']
        image_file_id ={}
        for img_info in images:
            file_name_no_suffix = img_info['file_name'].split('.')[0]
            image_file_id[file_name_no_suffix] = img_info['id']
        for pic_info in jdict:
            pic_info['image_id'] = image_file_id[pic_info['image_name']]

        with open(pred_json, 'w') as f:
            json.dump(jdict, f, indent=4)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            for conf_t in last_conf:
                pred_json_c = select_score_from_json(pred_json, score_thresh=conf_t)

                pred = anno.loadRes(pred_json_c)  # init predictions api
                eval = COCOeval(anno, pred, 'bbox')
                if is_coco:
                    eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
                eval.evaluate()
                eval.accumulate()
                eval.summarize()
                map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)

                if visual_matched:
                    with open(data[task], 'r') as f:
                        img0 = f.readline()
                    img_prefix = os.path.split(img0)[0]
                    print_log(
                        f"##########################\nnow we collect and visual result with iou thresh of "
                        f"0.5 and conf thresh of {conf_t}", logger=logger)

                    visual_return(eval, anno, save_dir, img_prefix, class_area=None, score_thresh=conf_t,
                                  logger=logger)

        except Exception as e:
            logger.info(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    logger.info(f"Results saved to {save_dir}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i, 0]

    det_result = (mp, mr, map50, map, (loss.cpu() / len(dataloader)).tolist())
    return det_result, maps, t


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', type=str,  nargs='+', default=ROOT / 'yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--cfg', default=None, help='model yaml path(s)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--bgr', type=int, default=1, help='if 1 bgr,0 gray')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'results/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--compute-loss', type=str, default=None, help='loss name used')
    parser.add_argument('--div-area', nargs='+', type=int, default=None, help='area of pixel to divide')
    parser.add_argument('--last-conf', nargs='+', type=float, default=0.1, help='last conf thresh after nms')
    parser.add_argument('--visual-matched', action='store_true', help='match the gt and pred and visual')
    parser.add_argument('--filter-str', type=str, default='', help='filter and select the image name with string')
    parser.add_argument('--ignore-bkg', action='store_true', help='filter and image of background')
    parser.add_argument('--loss-num', type=int, default=3, help='loss num of class , detect or seg')
    parser.add_argument('--train-val-filter', action='store_true', help='filter first use the classify head')
    parser.add_argument('--val-train', action='store_true', help='valuate the train dataset')
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    return opt


def main(opt):

    # Data
    data_file = Path(opt.data).stem
    opt.data = check_dataset(opt.data)  # check
    temp_weights = opt.weights
    weight_list = []
    # for temp_weight in temp_weights:
    #     if os.path.isdir(temp_weight):
    #         for weight in glob(f'{temp_weight}/*.pt'):
    #             # if 'zjs_' in weight and 'merge' not in weight:
    #             #     weight_list.append(weight)
    #             weight_list.append(weight)
    #     else:
    #         weight_list.append(opt.weights)
    for temp_weight in temp_weights:
        if os.path.isdir(temp_weight):
            weight_list += glob(f'{temp_weight}/**/*.pt', recursive=True)
        else:
            weight_list.append(temp_weight)

    for opt.weights in weight_list:
        opt.device = select_device(opt.device, batch_size=opt.batch_size)
        # Directories
        if 'weights' in opt.weights.split('/'):
            weight_file_name = '_'.join(opt.weights.split('/')[-4:-2]) + '_'
        else:
            weight_file_name = ''
        weight_file_name += str(Path(opt.weights).stem)

        opt.save_dir = increment_path(Path(opt.project)/ data_file / weight_file_name / opt.name, exist_ok=opt.exist_ok)  # increment run
        (opt.save_dir / 'labels' if opt.save_txt else opt.save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        opt.logger = set_logging(name=FILE.stem, filename=Path(Path(opt.save_dir) / 'val.log'))

        print_args(FILE.stem, opt, logger=opt.logger)
        opt.logger.info('#############################################')
        opt.logger.info(f'now we val {opt.weights}:\n saved in {opt.save_dir}')
        # Load model
        opt.model = attempt_load(opt.weights, map_location=opt.device, cfg=opt.cfg, nc=int(opt.data['nc']), logger=opt.logger)
        opt.model = opt.model.module if hasattr(opt.model, 'module') else opt.model
        # for p in opt.model.parameters():
        #     p.requires_grad = False
        stride = int(opt.model.stride.max()) if hasattr(opt.model, 'stride') else 32
        opt.model.stride = stride
        opt.model.names = opt.data['names']  # get class names
        # if  not hasattr(opt.model, 'device'):
        opt.model.device = opt.device
        opt.model.train_val_filter = opt.train_val_filter
        # stride = 32  # grid size (max stride)
        # opt.imgsz = check_img_size(opt.imgsz, s=stride, logger=opt.logger)  # check image size

        if opt.compute_loss is not None:
            if isinstance(opt.compute_loss, str):
                opt.compute_loss = eval(opt.compute_loss)(opt.model, logger=opt.logger)


        task = opt.task if opt.task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        if opt.val_train and task != 'train':
            tasks = (task, 'train')
        else:
            tasks = (task,)
        for opt.task in tasks:
            opt.model.hyp['val_sample_portion']= 1
            dataset = LoadImagesAndLabels(opt.data[opt.task], opt.imgsz, opt.batch_size, opt.logger,
                                          hyp=opt.model.hyp,  # hyperparameters
                                          stride=int(stride),
                                          augment=True,
                                          rect=True,
                                          pad=.5,
                                          prefix=f'{opt.task}: ',
                                          is_bgr=opt.bgr,
                                          filter_str=opt.filter_str,
                                          filter_bkg=opt.ignore_bkg,
                                          select_class=select_class_tuple(opt.data))
            batch_size = min(opt.batch_size, len(dataset))
            opt.dataloader = DataLoader(dataset,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=4,
                          sampler=None,
                          pin_memory=True,
                          collate_fn=LoadImagesAndLabels.collate_fn)

            # run normally

            results, maps,times = run(**vars(opt))
            msg = 'task{task}:   Loss({loss})\n' \
                  'Detect: P({p:.3f})  R({r:.3f})  mAP@0.5({map50:.3f})  mAP@0.5:0.95({map:.3f})\n' \
                  'Time: inference({t_inf:.4f}ms/frame)  nms({t_nms:.4f}ms/frame)'.format(
                task=opt.task, loss=results[-1],
                p=results[0], r=results[1], map50=results[2], map=results[3],
                t_inf=times[1], t_nms=times[2])
            opt.logger.info(msg)
    print('validate done')

    torch.cuda.empty_cache()


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
