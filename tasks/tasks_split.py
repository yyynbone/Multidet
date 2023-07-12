import os
import random
from pathlib import Path
from threading import Thread
import numpy as np
from tqdm import tqdm
from glob import glob
from copy import deepcopy
import yaml
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader, distributed
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD, Adam, lr_scheduler
import torch.distributed as dist

from loss import *
from models  import (attempt_load, Model)
from dataloader import check_anchors, check_train_batch_size, LoadImagesAndLabels, get_dataset, SuffleLoader, SuffleDist_Sample
from utils.lr_schedule import *
from utils import ( check_dataset, check_img_size, check_suffix, colorstr, time_sync, init_seeds,
                    labels_to_class_weights, labels_to_image_weights, strip_optimizer, cal_flops, intersect_dicts,
                    Callback, de_parallel, torch_distributed_zero_first, EarlyStopping, ModelEMA, LossSaddle,
                    output_to_target, xywh2xyxy, xyxy2xywh, non_max_suppression_with_iof,
                    div_area_idx, process_batch,  ConfusionMatrix, det_calculate, det_coco_calculate,
                    select_class_tuple, save_one_txt, save_one_json,  plot_results, plot_images, plot_labels,
                    visual_match_pred, print_log, classify_match, save_object )

FILE = Path(__file__).resolve()


def before_train(hyp, opt, logger):
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze, = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze
    max_stride = 4
    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    print_log(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()), logger)

    # Save run settings
    if not evolve:
        with open(save_dir / 'hyp.yaml', 'w') as f:
            yaml.safe_dump(hyp, f, sort_keys=False)
        with open(save_dir / 'opt.yaml', 'w') as f:
            yaml.safe_dump(vars(opt), f, sort_keys=False)

    # Loggers
    # this will run in each local_rank
    data_dict =  check_dataset(data)  # check if None
    opt.data_dict =  data_dict
    train_path, val_path = data_dict['train'], data_dict['val']

    with open(opt.data_prefile, errors='ignore') as f:
        data_pre = yaml.safe_load(f)  # load hyps dict
    train_pre = data_pre['train']
    val_pre = data_pre['val']

    nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    if not hyp.get('use_BCE', True):
        nc += 1
    hyp['nc'] = nc
    hyp['names'] = names
    # assert len(names) == nc, f'{len(names)} names found for nc={nc} dataset in {data}'  # check


    if opt.bgr:
        ch_in = 3
    else:
        ch_in = 1
    if resume:
        ckpt = torch.load(weights, map_location='cpu')  # load checkpoint
        model = ckpt.get('ema', ckpt['model']).float()
        opt.cfg = ''
    else:
        model = Model(cfg, ch=ch_in, nc=nc, anchors=hyp.get('anchors'), logger=logger) # create
    # Image size
    # gs = int(model.stride.max()) if hasattr(model, 'stride') else max_stride
    # gs = max(gs, max_stride)  # grid size (max stride)
    gs = max_stride

    imgsz = opt.imgsz
    if isinstance(imgsz, int):
        imgsz = (imgsz, imgsz)  # h, w
    elif len(imgsz) == 1:
        imgsz = (imgsz[0], imgsz[0])
    imgsz = check_img_size(imgsz, gs, floor=gs * 2, logger=logger)  # verify imgsz is gs-multiple
    fs = cal_flops(de_parallel(model), imgsz)

    opt.imgsz = imgsz
    opt.gs = gs
    opt.hyp = hyp
    # Batch size
    if batch_size == -1:  # single-GPU only, estimate best batch size
        batch_size = check_train_batch_size(model, imgsz)

    # Train
    train_dataset, train_pre = get_dataset(train_path,
                                    train_pre,
                                    img_size=imgsz,
                                    batch_size=batch_size,
                                    logger=logger,
                                    single_cls=single_cls,
                                    bgr=opt.bgr,
                                    stride=gs,
                                    pad=.0,
                                    prefix=colorstr('train: '),
                                    filter_str=opt.filter_str,
                                    filter_bkg=opt.ignore_bkg,
                                    select_class=select_class_tuple(data_dict))

    mlc = 0  # int(np.concatenate(train_dataset.labels, 0)[:, 0].max())  # max label class
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'

    batch_size = min(batch_size, len(train_dataset))

    # Val
    val_dataset, val_pre = get_dataset(val_path,
                                      val_pre,
                                      img_size=imgsz,
                                      batch_size=batch_size,
                                      logger=logger,
                                      single_cls=single_cls,
                                      bgr=opt.bgr,
                                      stride=gs,
                                      pad=.0,
                                      prefix=colorstr('val: '),
                                      filter_str=opt.filter_str,
                                      filter_bkg=opt.ignore_bkg,
                                      select_class=select_class_tuple(data_dict))



    print_log(f'Image sizes {imgsz} train, {imgsz} val\n'
              f'now with size of {imgsz}, {fs:.2f} GFLOPS\n'
              f'Each epoch batch_size is {batch_size},iter is {len(train_dataset) / batch_size}\n'
              f"Logging results to {colorstr('bold', save_dir)}\n", logger)

    return  opt, train_dataset, train_pre, val_dataset, val_pre

def train(opt, logger, device, train_data, train_pre, val_data, val_pre,  local_rank=-1, node=-1, world_size=1):

    save_dir, evolve, data, hyp, weights, cfg, resume, noval, nosave, workers, freeze = Path(opt.save_dir), opt.evolve,\
            opt.data, opt.hyp, opt.weights, opt.cfg, opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze

    epochs, batch_size, single_cls, imgsz, gs, data_dict = opt.epochs, opt.batch_size, opt.single_cls, \
                                                  opt.imgsz, opt.gs, opt.data_dict

    # Directories
    w = save_dir / 'weights'  # weights dir
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'
    best_pt, best_rt = w / 'best_precision.pt', w / 'best_recall.pt'
    best_p, best_r = 0., 0.
    start_epoch, best_fitness = 0, 0.0
    # Config
    plots = not opt.evolve  # create plots
    cuda = device.type != 'cpu'


    # callback only in local_rank = 0
    if node in [-1, 0] and local_rank in [-1, 0]:
        callbacks = Callback(save_dir, opt, logger=logger)  # callback instance
    else:
        callbacks =None

        # Model
    check_suffix(weights, '.pt')  # check weights
    pretrained = weights.endswith('.pt')
    if opt.bgr:
        ch_in = 3
    else:
        ch_in = 1
    nc = hyp['nc']
    if pretrained:

        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        model = Model(cfg or ckpt['model'].yaml, ch=ch_in, nc=nc, anchors=hyp.get('anchors'), logger=logger).to(
            device)  # create

        # exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else active_layer # exclude keys
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys

        # checkpoint state_dict as FP32
        # csd =  ckpt['ema'].float().state_dict() if 'ema' in ckpt.keys()  else ckpt['model'].float().state_dict()
        if isinstance(ckpt, dict):
            if ckpt.get('ema', None) is not None:
                csd = ckpt['ema'].float().state_dict()
                csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
            elif ckpt.get('model', None) is not None:
                csd = ckpt['model'].float().state_dict()
                csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
            else:
                csd = ckpt['state_dict']
                csd = intersect_dicts(csd, model.state_dict(), exclude=exclude, key_match=False)  # intersect
        else:
            csd = intersect_dicts(ckpt, model.state_dict(), exclude=exclude, key_match=False)  # intersect
        model.load_state_dict(csd, strict=False)  # load
        print_log(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}', logger)  # report

    else:
        model = Model(cfg, ch=ch_in, nc=nc, anchors=hyp.get('anchors'), logger=logger).to(device)  # create
    # print('model load in: ', local_rank)
    # calculate class_weight
    train_dataset = LoadImagesAndLabels(train_pre,
                                        train_data,
                                        logger=logger,
                                        bkg_ratio=getattr(opt, "bkg_ratio", 0),
                                        obj_mask=getattr(opt, "train_obj_mask", 0)
                                        )
    train_dataset.cropped_imgsz =  getattr(opt, "train_cropped_imgsz", False)
    train_dataset.slide_crop = getattr(opt, "slide_crop", False)
    with torch_distributed_zero_first(local_rank):
        train_dataset.index_shuffle(local_rank)
    if node in [-1, 0] and local_rank in [-1, 0]:
        plot_labels(train_dataset.labels, names=select_class_tuple(data_dict), save_dir=save_dir, logger=logger)
    class_weights = labels_to_class_weights(train_dataset.labels, nc)
    hyp['cls_weight'] = class_weights
    opt.hyp = hyp
    if opt.rect:
        logger.warning('WARNING: --rect is incompatible with DataLoader shuffle, setting shuffle=False')
        shuffle = False
    else:
        shuffle = True
    sampler = None if opt.world_size == 1 else SuffleDist_Sample(train_dataset, shuffle=shuffle)
    # loader = DataLoader  # if opt.image_weights else InfiniteDataLoader  # only DataLoader allows for attribute updates
    batch_size = min(batch_size, len(train_dataset))
    train_loader = SuffleLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=shuffle and sampler is None,
                          num_workers=workers,
                          sampler=sampler,
                          pin_memory=True,
                          collate_fn=LoadImagesAndLabels.collate_fn)

    # if opt.image_weights:
    #     iw = labels_to_image_weights(train_dataset.labels, nc=nc, class_weights=class_weights / nc)  # image weights
    #     # eg:pic num is 10 , class_weights is (0.2,0.3,1,1,1,1,...)(because real class_num=2 (but to 80)
    #     # dataset.indices before is (0,1,2,...9)
    #     # but now is (2,3,2,3 ,4,2,0,0,3,2) like this, so some pic is not trained
    #     # dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)\
    #     # so modified to this:
    #     add_indices = random.choices(range(train_dataset.n), weights=iw, k=train_dataset.n)  # rand weighted idx
    #     train_dataset.indices = np.concatenate((train_dataset.indices, add_indices))
    print_log('train dataset loader done', logger)

    # Process 0
    if not resume and local_rank in [-1, 0]:
        # Anchors
        if not opt.noautoanchor:
            check_anchors(train_dataset, model=model, imgsz=imgsz,  thr=hyp['anchor_t'], logger=logger)

    # Valloader
    val_dataset = LoadImagesAndLabels(val_pre,
                                      val_data,
                                      logger=logger,
                                      bkg_ratio=0,
                                      obj_mask=getattr(opt, "val_obj_mask", 0)
                                      )
    val_dataset.cropped_imgsz =  getattr(opt, "val_cropped_imgsz", False)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=workers,
                            sampler=None,
                            pin_memory=True,
                            collate_fn=LoadImagesAndLabels.collate_fn)

    print_log(f"Epoch train image shape is {train_dataset[0][0].shape}, should input as (ch, y, x)", logger)

    # Freeze
    task_dict = {1: 'encoder', 2: 'neck', 3: 'head'}
    freeze = list(range(freeze))

    neck_from_idx = model.neck_from
    head_from_idx = model.head_from

    para_idx = [[i for i in range(0, neck_from_idx)],
                [i for i in range(neck_from_idx, head_from_idx)],
                [i for i in range(head_from_idx, len(model.model) + 1)],
                ]

    if not opt.task:  # [encoder,det,da_seg,l1_seg]
        pass
    else:
        for i in range(1, len(para_idx)):
            if not i in opt.task:
                print_log(f'freeze {task_dict[i]}', logger)
                freeze += para_idx[i - 1]
    # active_layer = [f'model.{x}.' for x in range(len(model.layer)) if x not in freeze]
    # freeze = [f'model.{x}.' for x in range(freeze)]  # layers to freeze

    freeze = [f'model.{x}.' for x in set(freeze)]

    model.freeze = freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print_log(f'freezing {k}', logger)
            v.requires_grad = False

    # with torch_distributed_zero_first(local_rank):
    #     show_model_param(model,'origin')

    # set model.train_val_filter
    model.train_val_filter = opt.train_val_filter
    g0, g1, g2 = [], [], []  # optimizer parameter groups
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g1.append(v.weight)

    if opt.adam:
        optimizer = Adam(g0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = SGD(g0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': g1, 'weight_decay': hyp['weight_decay']})  # add g1 with weight_decay
    optimizer.add_param_group({'params': g2})  # add g2 (biases)
    print_log(f"{colorstr('optimizer:')} {type(optimizer).__name__} with parameter groups "
              f"{len(g0)} weight, {len(g1)} weight (no decay), {len(g2)} bias", logger)
    del g0, g1, g2

    # EMA
    ema = ModelEMA(model)

    # Resume

    if pretrained and len(csd) == len(model.state_dict()):
        # Optimizer
        if ckpt['optimizer'] is not None and resume:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']
            print_log('now load optimizer param done', logger)

        # EMA
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']
            print_log('now load ema state dict param done', logger)

        # Epochs
        # start_epoch = 0
        if resume:
            print_log(f'Resuming training from {opt.weights}', logger)
            ckpt['epoch'] = ckpt.get('epoch', -1)
            start_epoch = ckpt['epoch'] + 1
            assert start_epoch > 0, f'{weights} training to {epochs} epochs is finished, nothing to resume.'
        if epochs < start_epoch:
            print_log(f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} more epochs.",
                      logger)
            epochs += ckpt['epoch']  # finetune additional epochs
        del ckpt, csd
    opt.best_fitness = best_fitness
    if hasattr(de_parallel(model), 'det_head_idx'):
        nl = de_parallel(model).model[de_parallel(model).det_head_idx].nl  # number of detection layers (to scale hyps)
        hyp['box'] *= 3 / nl  # scale to layers
        hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
        # hyp['obj'] *= (max(imgsz) / 640) ** 2 * 3 / nl  # scale to image size and layers
        hyp['label_smoothing'] = opt.label_smoothing



    init_seeds(1 + node)
    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    print_log(f"Scaled weight_decay = {hyp['weight_decay']}", logger)

    # DP mode
    if cuda and node == -1 and torch.cuda.device_count() > 1:
        logger.warning('WARNING: DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n')
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  modify
        # model = BalancedDataParallel(32,model,dim =0).to(device)
        model = torch.nn.DataParallel(model)
        #  modify  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

    # SyncBatchNorm
    if opt.sync_bn and cuda and node != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        print_log('Using SyncBatchNorm()', logger)

    nb = len(train_loader)  # number of batches

    # DDP mode
    if cuda and node != -1:
        try:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        except:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    model.nc = hyp['nc']
    model.names = hyp['names']
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = torch.from_numpy(hyp['cls_weight']) * model.nc  # attach class weights


    # Scheduler
    if opt.linear_lr:
        lf = linear(1, hyp['lrf'], epochs)  # linear

    elif hyp['lrf'] < 0.1 and epochs > 1000:
        lf = soft_cos_log(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)


    nw = max(round(hyp['warmup_epochs'] * nb), 100*32/batch_size)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1
    maps = np.zeros(model.nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    stopper = EarlyStopping(patience=opt.patience, logger=logger)
    # lr_coeft = torch.tensor(1.0, device=device)
    stop = False
    broadcast_list = [stop, 1.]  # stop, lr_coeft
    out_saddle = LossSaddle(patience=opt.loss_supervision, logger=logger)

    compute_loss = eval(opt.loss)(model, logger=logger)  # init loss class
    loss_num = compute_loss.loss_num
    pos_schedule = eval(hyp.get("pos_schedule", "linear"))(hyp.get('lr_start_pos_weight', 0),
                                                           hyp.get('max_pos_weight', 10), epochs)

    print_log(f'out_saddle is {out_saddle}\n stopper is {stopper} \n '
              f'Using {train_loader.num_workers} dataloader workers per gpu \n'
              f'compute_loss is {compute_loss} \n pos_schedule is {pos_schedule}\n'
              f'Starting training for {epochs} epochs...', logger)

    if opt.val_first:
        start_epoch -= 1

    # Start training
    t0 = time.time()

    for epoch in range(start_epoch, epochs):  # epoch ---------------------------------------------------------
        # print('before epoch after broadcast is', broadcast_list, local_rank)
        lr_coeft = broadcast_list[1]
        # print('before epoch after broadcast is', lr_coeft, local_rank)
        if not opt.val_first:
            model.train()
            # Update image weights (optional, single-GPU only)
            if opt.image_weights:
                cw = model.class_weights * (1 - maps) ** 2 / model.nc  # class weights
                iw = labels_to_image_weights(train_loader.dataset.labels, nc=model.nc, class_weights=cw)  # image weights
                train_dataset.indices = random.choices(range(len(train_dataset)), weights=iw,
                                                       k=len(train_dataset))  # rand weighted idx
             
            # Update mosaic border (optional)
            # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
            # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders
            if  epoch%getattr(opt, 'shuffle_epoch', 100)==0:
                print_log(f'now we shuffle index in {epoch}', logger)
                train_loader.shuffle_index(local_rank)
            # sampler = None if opt.world_size == 1 else distributed.DistributedSampler(train_dataset, shuffle=shuffle)
            # loader = DataLoader  # if opt.image_weights else InfiniteDataLoader  # only DataLoader allows for attribute updates
            # train_loader = loader(train_dataset,
            #                       batch_size=batch_size,
            #                       shuffle=shuffle and sampler is None,
            #                       num_workers=workers,
            #                       num_workers=workers,
            #                       sampler=sampler,
            #                       pin_memory=True,
            #                       collate_fn=LoadImagesAndLabels.collate_fn)
            # print('shuffle_index load in: ', local_rank)
            if node != -1:
                train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(train_loader)
            nb = len(train_loader)  # number of batches
            # print(len(train_dataset), nb, batch_size, train_loader.batch_size, train_loader.batch_sampler)
            # print(logger.handlers)
            # if node in [-1,0] and local_rank in [-1,0]:
            # if len(logger.handlers):
            if logger.level==20:
                print_log(f"{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}", logger)
                pbar = tqdm(pbar, total=nb, ncols=200, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
            optimizer.zero_grad()

            mloss = torch.zeros(loss_num, device=device)  # mean losses

            if hasattr(compute_loss, 'pos_weight'):
                # if compute_loss.pos_weight is None:
                #     ft_num = targets[0].shape[0] - targets[0].sum()
                #     label_pos_weight = torch.tensor([ft_num / max(targets[0].sum(), 1)])
                epoch_pos_w = round(pos_schedule(epoch), 2)
                if epoch_pos_w > compute_loss.label_pos_weight:
                    compute_loss.pos_weight = torch.tensor([epoch_pos_w])
                print_log(
                    f'now epoch_pos_w is {epoch_pos_w}, ClassifyLoss label_pos_weight is {compute_loss.label_pos_weight}, ClassifyLoss pos_weight is {compute_loss.pos_weight}',
                    logger)

            for i, (imgs, targets, paths) in pbar:  # batch -------------------------------------------
                # class_label, labels_out, seg_img = targets
                targets = [t.to(device) if t is not None else t for t in targets]
                # targets = targets.to(device)
                # if i > 1:
                #     break
                ni = i + nb * epoch  # number integrated batches (since train start)
                imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0

                # Warmup
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                    accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                    # optimizer.param_groups 每个iter都自动更新，即 enumerate(optimizer.param_groups) 只会有当前iter的g0,g1,g2 信息
                    for j, x in enumerate(optimizer.param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(ni, xi,
                                            [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

                # add lr grows up when in saddle point
                # 其实这是每个iter都会做这步，如有33个iter， x['lr']最后会变成 a*(lr_coeft**33),故设置每5个iter增加一次
                # if i%(nb//3) == 0: #每个epoch 4 次
                if i == 0:
                    for j, x in enumerate(optimizer.param_groups):
                        x['lr'] *= lr_coeft
                        if j == 2 and ni <= nw:
                            x['lr'] = min(x['lr'], hyp['warmup_bias_lr'])
                        else:
                            x['lr'] = min(x['lr'], x['initial_lr'])

                # if lr_coeft >1. :
                #     loggerprint_log(f"in epoch {epoch}, lr_coefficient is {lr_coeft}, lr is {x['lr']}")

                # Multi-scale
                if opt.multi_scale:
                    sz = random.randrange(max(imgsz) * 0.5, max(imgsz) * 1.5 + gs) // gs * gs  # size
                    sf = sz / max(imgs.shape[2:])  # scale factor
                    if sf != 1:
                        ns = [math.ceil(x * sf / gs) * gs for x in
                              imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                        imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

                # Forward
                with amp.autocast(enabled=cuda):
                    head_out = model(imgs)  # forward
                    det_pred = head_out[de_parallel(model).out_det_from]
                    loss, loss_items = compute_loss(det_pred, targets)  # loss scaled by batch_size
                    if node != -1:
                        loss *= world_size  # gradient averaged between devices in DDP mode
                    if opt.quad:
                        loss *= 4.

                # Backward

                scaler.scale(loss).backward()
                # Optimize
                # each iter batchsize must bigger then 64, eg：if bs is 128, accumulate=1 , if bs is 32, accumulate=2
                if ni - last_opt_step >= accumulate:
                    # clip the grad, to avoid the grad eruption(nan), clip_grad_norm_ or clip_grad_value_
                    # Divides ("unscales") the optimizer's gradient tensors by the scale factor，
                    # 这里统计并存入dict, optimizer_state["found_inf_per_device"]，check inf的数量，无则为0
                    scaler.unscale_(optimizer)
                    # # calculate grad norm,so that we can set max_norm
                    # parameters = [p for p in model.parameters() if p.grad is not None]
                    # total_norm = torch.norm(
                    #     torch.stack([torch.norm(p.grad.detach(), 2).to(device) for p in parameters]), 2)
                    # print(total_norm)
                    # total_norm in (100,200)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20)
                    # print(f"now optimizer step, ni{ni} - last_opt_step{last_opt_step} >="
                    # "accumulate{accumulate}, nbs is {nbs}, batch_size is {batch_size}")

                    # optimizer.step  # If inf/NaN gradients are found, `optimizer.step()`
                    # is skipped to avoid corrupting the params,使用self._maybe_opt_step，
                    scaler.step(optimizer)
                    # print("optimizer._step_count is ", optimizer._step_count)
                    # 前5个iter ,一直为0，第6个开始为1，然后开始累加  #由上分析知，梯度含inf值
                    scaler.update()
                    optimizer.zero_grad()
                    if node in [-1, 0] and local_rank in [-1, 0]:
                        ema.update(model)
                        # if i % 20 == 19:
                        #     show_model_param(ema.ema,f'after_{epoch}_{i}')
                    last_opt_step = ni

                # change callbacks keys
                if i == 0 and node in [-1, 0] and local_rank in [-1, 0]:
                    # loss_num = loss_items.size()[0] if len(loss_items.size()) else 1:
                    if loss_num == 3:
                        par_msg = '%10s' * 7 % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'labels', 'img_size')
                    elif loss_num == 4:
                        par_msg = '%10s' * 8 % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'seg', 'labels', 'img_size')
                        callbacks.keys = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',
                                          'train/seg_loss'  # train loss
                                          'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5',
                                          'metrics/mAP_0.5:0.95',  # metrics
                                          'val/box_loss', 'val/obj_loss', 'val/cls_loss', 'val/seg_loss'  # val loss
                                                                                          'x/lr0', 'x/lr1',
                                          'x/lr2']  # params
                    else:
                        callbacks.keys = ['train/cls_loss',  # train loss
                                          'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5',
                                          'metrics/mAP_0.5:0.95',  # metrics
                                          'val/cls_loss',  # val loss
                                          'x/lr0', 'x/lr1', 'x/lr2']  # params
                        par_msg = '%10s' * 7 % ('Epoch', 'gpu_mem', 'cls', 'all_pic', 'obj', 'labels', 'img_size')
                    print_log(par_msg, logger)
                # Log
                if node in [-1, 0] and local_rank in [-1, 0]:
                    mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                    mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                    if loss_num == 1:
                        pbar.set_description(('%10s' * 2 + '%10.4g' * (loss_num + 4)) % (
                            f'{epoch}/{epochs - 1}', mem, *mloss, targets[0].shape[0], targets[0].sum(),
                            targets[1].shape[0], imgs.shape[-1]))
                    else:
                        pbar.set_description(('%10s' * 2 + '%10.4g' * (2 + loss_num)) % (
                            f'{epoch}/{epochs - 1}', mem, *mloss, targets[1].shape[0], imgs.shape[-1]))

                    callbacks.on_train_batch_end(ni, model, imgs, targets, paths, plots, opt.sync_bn)

            # Scheduler
            lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
            # print("now scheduler step means lr_scheduler step")
            scheduler.step()
            if node in [-1, 0] and local_rank in [-1, 0] and opt.loss_supervision:
                lr_coeft = out_saddle(epoch=epoch, loss=loss.item())
                if lr_coeft > 1:
                    print_log(f'now in epoch {epoch}, lr coefficient is:{lr_coeft}', logger)

        if node in [-1, 0] and local_rank in [-1, 0]:
            if opt.val_train:
                results, _, times = val(data_dict,
                                        batch_size=batch_size,
                                        imgsz=imgsz,
                                        model=ema.ema,  # de_parallel(model),
                                        single_cls=single_cls,
                                        verbose=True,
                                        dataloader=train_loader,
                                        save_dir=save_dir,
                                        plots=False,
                                        compute_loss=compute_loss,
                                        bgr=opt.bgr,
                                        div_area=opt.div_area,
                                        last_conf=opt.last_conf,
                                        loss_num=loss_num,
                                        logger=logger)

                loss_tuple = results[-1]
                msg = 'Valuate traindataset before ema update(model update) Epoch: [{0}]    Loss({loss})\n' \
                      'Detect: P({p:.3f})  R({r:.3f})  mAP@0.5({map50:.3f})  mAP@0.5:0.95({map:.3f})\n' \
                      'Time: inference({t_inf:.4f}s/frame)  nms({t_nms:.4f}s/frame)'.format(
                    epoch, loss=loss_tuple,
                    p=results[0], r=results[1], map50=results[2], map=results[3],
                    t_inf=times[1], t_nms=times[2])
                print_log(msg, logger)

            # mAP
            # callbacks.on_train_epoch_end(epoch=epoch)
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            if not noval or final_epoch:  # Calculate mAP
                results, maps, times = val(data_dict,
                                           batch_size=batch_size,
                                           imgsz=imgsz,
                                           model=ema.ema,  # de_parallel(model),
                                           single_cls=single_cls,
                                           verbose=True,
                                           dataloader=val_loader,
                                           save_dir=save_dir,
                                           plots=True,
                                           compute_loss=compute_loss,
                                           bgr=opt.bgr,
                                           div_area=opt.div_area,
                                           last_conf=opt.last_conf,
                                           loss_num=loss_num,
                                           logger=logger)
                if not opt.val_first:
                    loss_tuple = results[-1]
                    msg = 'Epoch: [{0}]    Loss({loss})\n' \
                          'Detect: P({p:.3f})  R({r:.3f})  mAP@0.5({map50:.3f})  mAP@0.5:0.95({map:.3f})\n' \
                          'Time: inference({t_inf:.4f}s/frame)  nms({t_nms:.4f}s/frame)'.format(
                        epoch, loss=loss_tuple,
                        p=results[0], r=results[1], map50=results[2], map=results[3],
                        t_inf=times[1], t_nms=times[2])
                    all_results = results[:4]
                    log_vals = list(mloss) + list(results[:4]) + list(loss_tuple) + lr
                    print_log(msg, logger)

                    # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
                    # fi = fitness(np.array(all_results).reshape(1, -1))
                    fi = all_results[3]
                    precision_fi, recall_fi = all_results[:2]

                    callbacks.on_fit_epoch_end(log_vals, epoch)
                    # Save model
                    if (not nosave) or (final_epoch and not evolve):  # if save
                        ckpt = {'epoch': epoch,
                                'best_fitness': fi,
                                'model': deepcopy(de_parallel(model)).half(),
                                'ema': deepcopy(ema.ema).half(),
                                'updates': ema.updates,
                                'optimizer': optimizer.state_dict(),
                                'date': datetime.now().isoformat()}
                        # Save last, best and delete
                        if torch.isnan(torch.tensor(loss_tuple)).sum():
                            torch.save(ckpt, w / 'nanerupt.pt')
                            print("warning now wo found a nan in loss in val")
                            break
                        else:
                            torch.save(ckpt, last)
                        if fi > best_fitness:
                            best_fitness = fi
                            torch.save(ckpt, best)
                            print_log(f'best epoch saved in Epoch {epoch}', logger)

                        if recall_fi > 0.93 and precision_fi > 0.9:
                            torch.save(ckpt, w / f'epoch{epoch}.pt')
                            print_log(f'a better recall epoch saved in Epoch {epoch}', logger)
                        if recall_fi > 0.8:
                            re_fi = precision_fi + (2 - 0.8) / (1 - 0.8 ** 2) * recall_fi
                            if re_fi > best_r:
                                best_r = re_fi
                                torch.save(ckpt, best_rt)
                                print_log(f'best recall epoch saved in Epoch {epoch}', logger)

                        pr_fi = 2.8 * precision_fi * recall_fi / np.maximum(1.8 * precision_fi + recall_fi, 1e-20)

                        if pr_fi > best_p and recall_fi > 0.6:
                            best_p = pr_fi
                            torch.save(ckpt, best_pt)
                            print_log(f'best precision epoch saved in Epoch {epoch}', logger)

                        if (epoch > 0) and (opt.save_period > 0) and (epoch % opt.save_period == 0):
                            torch.save(ckpt, w / f'epoch{epoch}.pt')
                        if (epoch < hyp['warmup_epochs'] + 3) and opt.save_warm:
                            torch.save(ckpt, w / f'epoch{epoch}.pt')
                        del ckpt
                        # callbacks.on_model_save(last, epoch, final_epoch, best_fitness, fi)

                    stop = stopper(epoch=epoch, fitness=fi)
                    plot_results(file=save_dir / 'results.csv')  # save results.png

        if opt.val_first:
            opt.val_first = False

        #####################stop and broadcast#######################
        # Stop Single-GPU
        if node == -1:
            if stop:
                break
        else:
            # Stop DDP
            if node == 0 and local_rank == 0:
                broadcast_list = [stop, lr_coeft]
                # print('broadcast is', broadcast_list, local_rank)
            # else:
            #     broadcast_list = [None, None]
            # print('broadcast is', broadcast_list, local_rank)
            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
            # Stop DPP
            # print('now after broadcast is', broadcast_list, local_rank)
            with torch_distributed_zero_first(local_rank):
                stop = broadcast_list[0]
                if stop:
                    break  # must break all DDP ranks
            ############################################

        # end epoch ---------------------------------------------
    # end training ------------------------------------------------
    if node in [-1, 0] and local_rank in [-1, 0]:
        print_log(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.', logger)
        for f in glob(f'{w}/*.pt'):
            if os.path.exists(f):
                strip_optimizer(f)  # strip optimizers
                if f in (best, best_pt, best_rt):
                    print_log(f'\nValidating {f}...', logger)
                    results, _, _ = val(data_dict,
                                        batch_size=batch_size * 2,
                                        imgsz=imgsz,
                                        model=attempt_load(f, device).half(),
                                        iou_thres=0.60,  # best pycocotools results at 0.65
                                        single_cls=single_cls,
                                        dataloader=val_loader,
                                        save_dir=save_dir,
                                        verbose=True,
                                        plots=True,
                                        compute_loss=compute_loss,
                                        bgr=opt.bgr,
                                        div_area=opt.div_area,
                                        last_conf=opt.last_conf,
                                        loss_num=loss_num,
                                        logger=logger)  # val best model with plots

        callbacks.on_train_end(best, plots)
        print_log(f"Results saved to {colorstr('bold', save_dir)}", logger)

    torch.cuda.empty_cache()
    return results

def kd_train(opt, logger, device, train_data, train_pre, val_data, val_pre,  local_rank=-1, node=-1, world_size=1):

    save_dir, evolve, data, hyp, weights, cfg, resume, noval, nosave, workers, freeze = Path(opt.save_dir), opt.evolve,\
            opt.data, opt.hyp, opt.weights, opt.cfg, opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze

    epochs, batch_size, single_cls, imgsz, gs, data_dict = opt.epochs, opt.batch_size, opt.single_cls, \
                                                  opt.imgsz, opt.gs, opt.data_dict

    teacher_net = attempt_load(opt.teacher, device)
    temperature = opt.temperature
    # Directories
    w = save_dir / 'weights'  # weights dir
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'
    best_pt, best_rt = w / 'best_precision.pt', w / 'best_recall.pt'
    best_p, best_r = 0., 0.
    start_epoch, best_fitness = 0, 0.0
    # Config
    plots = not opt.evolve  # create plots
    cuda = device.type != 'cpu'


    # callback only in local_rank = 0
    if node in [-1, 0] and local_rank in [-1, 0]:
        callbacks = Callback(save_dir, opt, logger=logger)  # callback instance
    else:
        callbacks =None

        # Model
    check_suffix(weights, '.pt')  # check weights
    pretrained = weights.endswith('.pt')
    if opt.bgr:
        ch_in = 3
    else:
        ch_in = 1
    nc = hyp['nc']
    if pretrained:

        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        model = Model(cfg or ckpt['model'].yaml, ch=ch_in, nc=nc, anchors=hyp.get('anchors'), logger=logger).to(
            device)  # create

        # exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else active_layer # exclude keys
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys

        # checkpoint state_dict as FP32
        # csd =  ckpt['ema'].float().state_dict() if 'ema' in ckpt.keys()  else ckpt['model'].float().state_dict()
        if isinstance(ckpt, dict):
            if ckpt.get('ema', None) is not None:
                csd = ckpt['ema'].float().state_dict()
                csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
            elif ckpt.get('model', None) is not None:
                csd = ckpt['model'].float().state_dict()
                csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
            else:
                csd = ckpt['state_dict']
                csd = intersect_dicts(csd, model.state_dict(), exclude=exclude, key_match=False)  # intersect
        else:
            csd = intersect_dicts(ckpt, model.state_dict(), exclude=exclude, key_match=False)  # intersect
        model.load_state_dict(csd, strict=False)  # load
        print_log(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}', logger)  # report

    else:
        model = Model(cfg, ch=ch_in, nc=nc, anchors=hyp.get('anchors'), logger=logger).to(device)  # create

    # calculate class_weight
    train_dataset = LoadImagesAndLabels(train_pre,
                                        train_data,
                                        logger=logger,
                                        bkg_ratio=getattr(opt, "bkg_ratio", 0),
                                        obj_mask=getattr(opt, "train_obj_mask", 0)
                                        )
    class_weights = labels_to_class_weights(train_dataset.labels, nc)
    hyp['cls_weight'] = class_weights
    opt.hyp = hyp
    if opt.rect:
        logger.warning('WARNING: --rect is incompatible with DataLoader shuffle, setting shuffle=False')
        shuffle = False
    else:
        shuffle = True
    sampler = None if opt.world_size == 1 else distributed.DistributedSampler(train_dataset, shuffle=shuffle)
    loader = DataLoader  # if opt.image_weights else InfiniteDataLoader  # only DataLoader allows for attribute updates
    train_loader = loader(train_dataset,
                          batch_size=batch_size,
                          shuffle=shuffle and sampler is None,
                          num_workers=workers,
                          sampler=sampler,
                          pin_memory=True,
                          collate_fn=LoadImagesAndLabels.collate_fn)

    # if opt.image_weights:
    #     iw = labels_to_image_weights(train_dataset.labels, nc=nc, class_weights=class_weights / nc)  # image weights
    #     # eg:pic num is 10 , class_weights is (0.2,0.3,1,1,1,1,...)(because real class_num=2 (but to 80)
    #     # dataset.indices before is (0,1,2,...9)
    #     # but now is (2,3,2,3 ,4,2,0,0,3,2) like this, so some pic is not trained
    #     # dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)\
    #     # so modified to this:
    #     add_indices = random.choices(range(train_dataset.n), weights=iw, k=train_dataset.n)  # rand weighted idx
    #     train_dataset.indices = np.concatenate((train_dataset.indices, add_indices))
    print_log('train dataset loader done', logger)
    batch_size = min(batch_size, len(train_dataset))
    # Process 0
    if not resume and local_rank in [-1, 0]:
        # Anchors
        if not opt.noautoanchor:
            check_anchors(train_dataset, model=model, imgsz=imgsz, thr=hyp['anchor_t'], logger=logger)

    # Valloader
    val_dataset = LoadImagesAndLabels(val_pre,
                                      val_data,
                                      logger=logger,
                                      bkg_ratio=0,
                                      obj_mask=getattr(opt, "val_obj_mask", 0)
                                      )
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=workers,
                            sampler=None,
                            pin_memory=True,
                            collate_fn=LoadImagesAndLabels.collate_fn)

    print_log(f"Epoch train image shape is {train_dataset[0][0].shape}, should input as (ch, y, x)", logger)

    # Freeze
    task_dict = {1: 'encoder', 2: 'neck', 3: 'head'}
    freeze = list(range(freeze))

    neck_from_idx = model.neck_from
    head_from_idx = model.head_from

    para_idx = [[i for i in range(0, neck_from_idx)],
                [i for i in range(neck_from_idx, head_from_idx)],
                [i for i in range(head_from_idx, len(model.model) + 1)],
                ]

    if not opt.task:  # [encoder,det,da_seg,l1_seg]
        pass
    else:
        for i in range(1, len(para_idx)):
            if not i in opt.task:
                print_log(f'freeze {task_dict[i]}', logger)
                freeze += para_idx[i - 1]
    # active_layer = [f'model.{x}.' for x in range(len(model.layer)) if x not in freeze]
    # freeze = [f'model.{x}.' for x in range(freeze)]  # layers to freeze

    freeze = [f'model.{x}.' for x in set(freeze)]

    model.freeze = freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print_log(f'freezing {k}', logger)
            v.requires_grad = False

    # with torch_distributed_zero_first(local_rank):
    #     show_model_param(model,'origin')

    # set model.train_val_filter
    model.train_val_filter = opt.train_val_filter
    g0, g1, g2 = [], [], []  # optimizer parameter groups
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g1.append(v.weight)

    if opt.adam:
        optimizer = Adam(g0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = SGD(g0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': g1, 'weight_decay': hyp['weight_decay']})  # add g1 with weight_decay
    optimizer.add_param_group({'params': g2})  # add g2 (biases)
    print_log(f"{colorstr('optimizer:')} {type(optimizer).__name__} with parameter groups "
              f"{len(g0)} weight, {len(g1)} weight (no decay), {len(g2)} bias", logger)
    del g0, g1, g2

    # EMA
    ema = ModelEMA(model)

    # Resume

    if pretrained and len(csd) == len(model.state_dict()):
        # Optimizer
        if ckpt['optimizer'] is not None and resume:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']
            print_log('now load optimizer param done', logger)

        # EMA
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']
            print_log('now load ema state dict param done', logger)

        # Epochs
        # start_epoch = 0
        if resume:
            print_log(f'Resuming training from {opt.weights}', logger)
            ckpt['epoch'] = ckpt.get('epoch', -1)
            start_epoch = ckpt['epoch'] + 1
            assert start_epoch > 0, f'{weights} training to {epochs} epochs is finished, nothing to resume.'
        if epochs < start_epoch:
            print_log(f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} more epochs.",
                      logger)
            epochs += ckpt['epoch']  # finetune additional epochs
        del ckpt, csd
    opt.best_fitness = best_fitness
    if hasattr(de_parallel(model), 'det_head_idx'):
        nl = de_parallel(model).model[de_parallel(model).det_head_idx].nl  # number of detection layers (to scale hyps)
        hyp['box'] *= 3 / nl  # scale to layers
        hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
        # hyp['obj'] *= (max(imgsz) / 640) ** 2 * 3 / nl  # scale to image size and layers
        hyp['label_smoothing'] = opt.label_smoothing



    init_seeds(1 + node)
    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    print_log(f"Scaled weight_decay = {hyp['weight_decay']}", logger)

    # DP mode
    if cuda and node == -1 and torch.cuda.device_count() > 1:
        logger.warning('WARNING: DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n')
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  modify
        # model = BalancedDataParallel(32,model,dim =0).to(device)
        model = torch.nn.DataParallel(model)
        #  modify  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

    # SyncBatchNorm
    if opt.sync_bn and cuda and node != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        print_log('Using SyncBatchNorm()', logger)

    nb = len(train_loader)  # number of batches

    # DDP mode
    if cuda and node != -1:
        try:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        except:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    model.nc = hyp['nc']
    model.names = hyp['names']
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = torch.from_numpy(hyp['cls_weight']) * model.nc  # attach class weights


    # Scheduler
    if opt.linear_lr:
        lf = linear(1, hyp['lrf'], epochs)  # linear

    elif hyp['lrf'] < 0.1 and epochs > 1000:
        lf = soft_cos_log(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)


    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1
    maps = np.zeros(model.nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    stopper = EarlyStopping(patience=opt.patience, logger=logger)
    # lr_coeft = torch.tensor(1.0, device=device)
    stop = False
    broadcast_list = [stop, 1.]  # stop, lr_coeft
    out_saddle = LossSaddle(patience=opt.loss_supervision, logger=logger)

    compute_loss = eval(opt.loss)(model, logger=logger)  # init loss class
    loss_num = compute_loss.loss_num
    pos_schedule = eval(hyp.get("pos_schedule", "linear"))(hyp.get('lr_start_pos_weight', 0),
                                                           hyp.get('max_pos_weight', 10), epochs)

    print_log(f'out_saddle is {out_saddle}\n stopper is {stopper} \n '
              f'Using {train_loader.num_workers} dataloader workers per gpu \n'
              f'compute_loss is {compute_loss} \n pos_schedule is {pos_schedule}\n'
              f'Starting training for {epochs} epochs...', logger)

    if opt.val_first:
        start_epoch -= 1

    # Start training
    t0 = time.time()

    for epoch in range(start_epoch, epochs):  # epoch ---------------------------------------------------------
        # print('before epoch after broadcast is', broadcast_list, local_rank)
        lr_coeft = broadcast_list[1]
        # print('before epoch after broadcast is', lr_coeft, local_rank)
        if not opt.val_first:
            model.train()
            # Update image weights (optional, single-GPU only)
            if opt.image_weights:
                cw = model.class_weights * (1 - maps) ** 2 / model.nc  # class weights
                iw = labels_to_image_weights(train_loader.dataset.labels, nc=model.nc, class_weights=cw)  # image weights
                train_loader.dataset.indices = random.choices(range(len(train_loader.dataset)), weights=iw,
                                                       k=len(train_loader.dataset))  # rand weighted idx
            else:
                train_loader.dataset.index_shuffle()
            # Update mosaic border (optional)
            # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
            # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

            if node != -1:
                train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(train_loader)

            # print(logger.handlers)
            # if node in [-1,0] and local_rank in [-1,0]:
            # if len(logger.handlers):
            if logger.level==20:
                print_log(f"{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}", logger)
                pbar = tqdm(pbar, total=nb, ncols=200, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
            optimizer.zero_grad()

            mloss = torch.zeros(loss_num, device=device)  # mean losses

            if hasattr(compute_loss, 'pos_weight'):
                # if compute_loss.pos_weight is None:
                #     ft_num = targets[0].shape[0] - targets[0].sum()
                #     label_pos_weight = torch.tensor([ft_num / max(targets[0].sum(), 1)])
                epoch_pos_w = round(pos_schedule(epoch), 2)
                if epoch_pos_w > compute_loss.label_pos_weight:
                    compute_loss.pos_weight = torch.tensor([epoch_pos_w])
                print_log(
                    f'now epoch_pos_w is {epoch_pos_w}, ClassifyLoss label_pos_weight is {compute_loss.label_pos_weight}, ClassifyLoss pos_weight is {compute_loss.pos_weight}',
                    logger)

            for i, (imgs, targets, paths) in pbar:  # batch -------------------------------------------
                # class_label, labels_out, seg_img = targets
                targets = [t.to(device) if t is not None else t for t in targets]
                # targets = targets.to(device)
                # if i > 1:
                #     break
                ni = i + nb * epoch  # number integrated batches (since train start)
                imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0

                # Warmup
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                    accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                    # optimizer.param_groups 每个iter都自动更新，即 enumerate(optimizer.param_groups) 只会有当前iter的g0,g1,g2 信息
                    for j, x in enumerate(optimizer.param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(ni, xi,
                                            [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

                # add lr grows up when in saddle point
                # 其实这是每个iter都会做这步，如有33个iter， x['lr']最后会变成 a*(lr_coeft**33),故设置每5个iter增加一次
                # if i%(nb//3) == 0: #每个epoch 4 次
                if i == 0:
                    for j, x in enumerate(optimizer.param_groups):
                        x['lr'] *= lr_coeft
                        if j == 2 and ni <= nw:
                            x['lr'] = min(x['lr'], hyp['warmup_bias_lr'])
                        else:
                            x['lr'] = min(x['lr'], x['initial_lr'])

                # if lr_coeft >1. :
                #     loggerprint_log(f"in epoch {epoch}, lr_coefficient is {lr_coeft}, lr is {x['lr']}")

                # Multi-scale
                if opt.multi_scale:
                    sz = random.randrange(max(imgsz) * 0.5, max(imgsz) * 1.5 + gs) // gs * gs  # size
                    sf = sz / max(imgs.shape[2:])  # scale factor
                    if sf != 1:
                        ns = [math.ceil(x * sf / gs) * gs for x in
                              imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                        imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

                # Forward
                with amp.autocast(enabled=cuda):
                    head_out = model(imgs)  # forward
                    det_pred = head_out[de_parallel(model).out_det_from]
                    loss, loss_items = compute_loss(det_pred, targets, teacher_net)  # loss scaled by batch_size

                    if teacher_net is not None:
                        loss_lambda = 0.5
                        te_out, te_pred = teacher_net(imgs)[teacher_net.out_det_from]
                        te_loss = teacher_loss(det_pred, te_pred, temperature)
                        loss = (1 - loss_lambda) * loss + loss_lambda * te_loss
                    if node != -1:
                        loss *= world_size  # gradient averaged between devices in DDP mode
                    if opt.quad:
                        loss *= 4.

                # Backward

                scaler.scale(loss).backward()
                # Optimize
                # each iter batchsize must bigger then 64, eg：if bs is 128, accumulate=1 , if bs is 32, accumulate=2
                if ni - last_opt_step >= accumulate:
                    # clip the grad, to avoid the grad eruption(nan), clip_grad_norm_ or clip_grad_value_
                    # Divides ("unscales") the optimizer's gradient tensors by the scale factor，
                    # 这里统计并存入dict, optimizer_state["found_inf_per_device"]，check inf的数量，无则为0
                    scaler.unscale_(optimizer)
                    # # calculate grad norm,so that we can set max_norm
                    # parameters = [p for p in model.parameters() if p.grad is not None]
                    # total_norm = torch.norm(
                    #     torch.stack([torch.norm(p.grad.detach(), 2).to(device) for p in parameters]), 2)
                    # print(total_norm)
                    # total_norm in (100,200)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20)
                    # print(f"now optimizer step, ni{ni} - last_opt_step{last_opt_step} >="
                    # "accumulate{accumulate}, nbs is {nbs}, batch_size is {batch_size}")

                    # optimizer.step  # If inf/NaN gradients are found, `optimizer.step()`
                    # is skipped to avoid corrupting the params,使用self._maybe_opt_step，
                    scaler.step(optimizer)
                    # print("optimizer._step_count is ", optimizer._step_count)
                    # 前5个iter ,一直为0，第6个开始为1，然后开始累加  #由上分析知，梯度含inf值
                    scaler.update()
                    optimizer.zero_grad()
                    if node in [-1, 0] and local_rank in [-1, 0]:
                        ema.update(model)
                        # if i % 20 == 19:
                        #     show_model_param(ema.ema,f'after_{epoch}_{i}')
                    last_opt_step = ni

                # change callbacks keys
                if i == 0 and node in [-1, 0] and local_rank in [-1, 0]:
                    # loss_num = loss_items.size()[0] if len(loss_items.size()) else 1:
                    if loss_num == 3:
                        par_msg = '%10s' * 7 % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'labels', 'img_size')
                    elif loss_num == 4:
                        par_msg = '%10s' * 8 % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'seg', 'labels', 'img_size')
                        callbacks.keys = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',
                                          'train/seg_loss'  # train loss
                                          'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5',
                                          'metrics/mAP_0.5:0.95',  # metrics
                                          'val/box_loss', 'val/obj_loss', 'val/cls_loss', 'val/seg_loss'  # val loss
                                                                                          'x/lr0', 'x/lr1',
                                          'x/lr2']  # params
                    else:
                        callbacks.keys = ['train/cls_loss',  # train loss
                                          'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5',
                                          'metrics/mAP_0.5:0.95',  # metrics
                                          'val/cls_loss',  # val loss
                                          'x/lr0', 'x/lr1', 'x/lr2']  # params
                        par_msg = '%10s' * 7 % ('Epoch', 'gpu_mem', 'cls', 'all_pic', 'obj', 'labels', 'img_size')
                    print_log(par_msg, logger)
                # Log
                if node in [-1, 0] and local_rank in [-1, 0]:
                    mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                    mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                    if loss_num == 1:
                        pbar.set_description(('%10s' * 2 + '%10.4g' * (loss_num + 4)) % (
                            f'{epoch}/{epochs - 1}', mem, *mloss, targets[0].shape[0], targets[0].sum(),
                            targets[1].shape[0], imgs.shape[-1]))
                    else:
                        pbar.set_description(('%10s' * 2 + '%10.4g' * (2 + loss_num)) % (
                            f'{epoch}/{epochs - 1}', mem, *mloss, targets[1].shape[0], imgs.shape[-1]))

                    callbacks.on_train_batch_end(ni, model, imgs, targets, paths, plots, opt.sync_bn)

            # Scheduler
            lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
            # print("now scheduler step means lr_scheduler step")
            scheduler.step()
            if node in [-1, 0] and local_rank in [-1, 0] and opt.loss_supervision:
                lr_coeft = out_saddle(epoch=epoch, loss=loss.item())
                if lr_coeft > 1:
                    print_log(f'now in epoch {epoch}, lr coefficient is:{lr_coeft}', logger)

        if node in [-1, 0] and local_rank in [-1, 0]:
            if opt.val_train:
                results, _, times = val(data_dict,
                                        batch_size=batch_size,
                                        imgsz=imgsz,
                                        model=ema.ema,  # de_parallel(model),
                                        single_cls=single_cls,
                                        verbose=True,
                                        dataloader=train_loader,
                                        save_dir=save_dir,
                                        plots=False,
                                        compute_loss=compute_loss,
                                        bgr=opt.bgr,
                                        div_area=opt.div_area,
                                        last_conf=opt.last_conf,
                                        loss_num=loss_num,
                                        logger=logger)

                loss_tuple = results[-1]
                msg = 'Valuate traindataset before ema update(model update) Epoch: [{0}]    Loss({loss})\n' \
                      'Detect: P({p:.3f})  R({r:.3f})  mAP@0.5({map50:.3f})  mAP@0.5:0.95({map:.3f})\n' \
                      'Time: inference({t_inf:.4f}s/frame)  nms({t_nms:.4f}s/frame)'.format(
                    epoch, loss=loss_tuple,
                    p=results[0], r=results[1], map50=results[2], map=results[3],
                    t_inf=times[1], t_nms=times[2])
                print_log(msg, logger)

            # mAP
            # callbacks.on_train_epoch_end(epoch=epoch)
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            if not noval or final_epoch:  # Calculate mAP
                results, maps, times = val(data_dict,
                                           batch_size=batch_size,
                                           imgsz=imgsz,
                                           model=ema.ema,  # de_parallel(model),
                                           single_cls=single_cls,
                                           verbose=True,
                                           dataloader=val_loader,
                                           save_dir=save_dir,
                                           plots=True,
                                           compute_loss=compute_loss,
                                           bgr=opt.bgr,
                                           div_area=opt.div_area,
                                           last_conf=opt.last_conf,
                                           loss_num=loss_num,
                                           logger=logger)
                if not opt.val_first:
                    loss_tuple = results[-1]
                    msg = 'Epoch: [{0}]    Loss({loss})\n' \
                          'Detect: P({p:.3f})  R({r:.3f})  mAP@0.5({map50:.3f})  mAP@0.5:0.95({map:.3f})\n' \
                          'Time: inference({t_inf:.4f}s/frame)  nms({t_nms:.4f}s/frame)'.format(
                        epoch, loss=loss_tuple,
                        p=results[0], r=results[1], map50=results[2], map=results[3],
                        t_inf=times[1], t_nms=times[2])
                    all_results = results[:4]
                    log_vals = list(mloss) + list(results[:4]) + list(loss_tuple) + lr
                    print_log(msg, logger)

                    # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
                    # fi = fitness(np.array(all_results).reshape(1, -1))
                    fi = all_results[3]
                    precision_fi, recall_fi = all_results[:2]

                    callbacks.on_fit_epoch_end(log_vals, epoch)
                    # Save model
                    if (not nosave) or (final_epoch and not evolve):  # if save
                        ckpt = {'epoch': epoch,
                                'best_fitness': fi,
                                'model': deepcopy(de_parallel(model)).half(),
                                'ema': deepcopy(ema.ema).half(),
                                'updates': ema.updates,
                                'optimizer': optimizer.state_dict(),
                                'date': datetime.now().isoformat()}
                        # Save last, best and delete
                        if torch.isnan(torch.tensor(loss_tuple)).sum():
                            torch.save(ckpt, w / 'nanerupt.pt')
                            print("warning now wo found a nan in loss in val")
                            break
                        else:
                            torch.save(ckpt, last)
                        if fi > best_fitness:
                            best_fitness = fi
                            torch.save(ckpt, best)
                            print_log(f'best epoch saved in Epoch {epoch}', logger)

                        if recall_fi > 0.93 and precision_fi > 0.9:
                            torch.save(ckpt, w / f'epoch{epoch}.pt')
                            print_log(f'a better recall epoch saved in Epoch {epoch}', logger)
                        if recall_fi > 0.8:
                            re_fi = precision_fi + (2 - 0.8) / (1 - 0.8 ** 2) * recall_fi
                            if re_fi > best_r:
                                best_r = re_fi
                                torch.save(ckpt, best_rt)
                                print_log(f'best recall epoch saved in Epoch {epoch}', logger)

                        pr_fi = 2.8 * precision_fi * recall_fi / np.maximum(1.8 * precision_fi + recall_fi, 1e-20)

                        if pr_fi > best_p and recall_fi > 0.6:
                            best_p = pr_fi
                            torch.save(ckpt, best_pt)
                            print_log(f'best precision epoch saved in Epoch {epoch}', logger)

                        if (epoch > 0) and (opt.save_period > 0) and (epoch % opt.save_period == 0):
                            torch.save(ckpt, w / f'epoch{epoch}.pt')
                        if (epoch < hyp['warmup_epochs'] + 3) and opt.save_warm:
                            torch.save(ckpt, w / f'epoch{epoch}.pt')
                        del ckpt
                        # callbacks.on_model_save(last, epoch, final_epoch, best_fitness, fi)

                    stop = stopper(epoch=epoch, fitness=fi)
                    plot_results(file=save_dir / 'results.csv')  # save results.png

        if opt.val_first:
            opt.val_first = False

        #####################stop and broadcast#######################
        # Stop Single-GPU
        if node == -1:
            if stop:
                break
        else:
            # Stop DDP
            if node == 0 and local_rank == 0:
                broadcast_list = [stop, lr_coeft]
                # print('broadcast is', broadcast_list, local_rank)
            # else:
            #     broadcast_list = [None, None]
            # print('broadcast is', broadcast_list, local_rank)
            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
            # Stop DPP
            # print('now after broadcast is', broadcast_list, local_rank)
            with torch_distributed_zero_first(local_rank):
                stop = broadcast_list[0]
                if stop:
                    break  # must break all DDP ranks
            ############################################

        # end epoch ---------------------------------------------
    # end training ------------------------------------------------
    if node in [-1, 0] and local_rank in [-1, 0]:
        print_log(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.', logger)
        for f in glob(f'{w}/*.pt'):
            if os.path.exists(f):
                strip_optimizer(f)  # strip optimizers
                if f in (best, best_pt, best_rt):
                    print_log(f'\nValidating {f}...', logger)
                    results, _, _ = val(data_dict,
                                        batch_size=batch_size * 2,
                                        imgsz=imgsz,
                                        model=attempt_load(f, device).half(),
                                        iou_thres=0.60,  # best pycocotools results at 0.65
                                        single_cls=single_cls,
                                        dataloader=val_loader,
                                        save_dir=save_dir,
                                        verbose=True,
                                        plots=True,
                                        compute_loss=compute_loss,
                                        bgr=opt.bgr,
                                        div_area=opt.div_area,
                                        last_conf=opt.last_conf,
                                        loss_num=loss_num,
                                        logger=logger)  # val best model with plots

        callbacks.on_train_end(best, plots)
        print_log(f"Results saved to {colorstr('bold', save_dir)}", logger)

    torch.cuda.empty_cache()
    return results

@torch.no_grad()
def val(data,
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
        iouv=(0.3, 0.95),
        iof_nms=True,
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
    # nc = len(model['names'])  if hasattr(model, 'names') else int(data['nc'])  # number of classes
    nc = model.nc if hasattr(model, 'nc') else int(data['nc']) # number of classes
    iouv = torch.linspace(iouv[0], iouv[1], int((iouv[1]-iouv[0])/0.05)+1).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)

    names =  {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else [])}
    for ci, cls in enumerate(data.get('names', [])):
        names[ci] = cls

    dt = [0.0, 0.0, 0.0]
    loss = torch.zeros(loss_num, device=device)  # mean losses
    jdict = []
    stats, cls_stats, cls_stats_indet = [], [], []
    div_class_num = 1
    if isinstance(div_area, (list, tuple)):
        div_class_num += len(div_area) + 1
    pbar = tqdm(dataloader, desc='validate detection', bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar

    for batch_i, (im, targets, paths) in enumerate(pbar):

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
            out = non_max_suppression_with_iof(out, conf_thres, iou_thres, labels=lb, multi_label=True, iof_nms=iof_nms)
            dt[2] += time_sync() - t3

            # Metrics
            for si, pred in enumerate(out):
                labels = targets[targets[:, 0] == si, 1:]
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []  # target class
                path = Path(paths[si])

                if len(pred) == 0:
                    if nl:
                        area_idx = torch.zeros(nl, div_class_num, dtype=torch.bool)
                        area = (labels[:, 2] * labels[:, 3]).cpu()
                        area_idx = div_area_idx(area, div_area, area_idx)
                        stats.append((torch.zeros(0, div_class_num, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls, area_idx))
                    continue

                # Predictions
                predn = pred.clone()   # predn is a pred from scaled image to origin size image
                # scale_coords(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

                # Evaluate
                if nl:
                    tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                    # scale_coords(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
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
                    save_one_txt(predn, save_conf, im.shape[2:4], file=save_dir / 'labels' / (path.stem + '.txt'))
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
                Thread(target=plot_images, args=(im, plot_tar, paths, f, names), daemon=False).start()
                f = save_dir / f'val_batch{batch_i}_pred.jpg'  # predictions
                Thread(target=plot_images, args=(im, plot_pred, paths, f, names), daemon=False).start()
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

    # Return results
    model.float()  # for training

    # Compute metrics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    cls_stats = [np.concatenate(x, 0) for x in zip(*cls_stats)]  # to numpy
    cls_stats_indet = [np.concatenate(x, 0) for x in zip(*cls_stats_indet)]  # to numpy


    if len(cls_stats):
        ps, rs , accus = classify_match(*cls_stats, conf_ts=last_conf, logger=logger)
        dt[2] += 0.
        det_result = (ps[1], rs[1], 0, accus[0], (loss.cpu() / len(dataloader)).tolist())
        # Print speeds
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        shape = (batch_size, ch_in, imgsz[0], imgsz[1])
        print_log(
            f'all image number is {seen}, which cost {sum(dt)}s\nSpeed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t, logger)
        return det_result, 0, t

    if len(cls_stats_indet):
        _, _, _ = classify_match(*cls_stats_indet, conf_ts=(getattr(model, 'class_conf', 0.4)), logger=logger)

    if len(stats) and stats[0].any():
        mp, mr, map50, map, ap, ap_class, p, r, = det_calculate(stats, div_area, nc, last_conf, save_dir, names, plots, verbose, logger)
        maps = np.zeros(nc) + map
        for i, c in enumerate(ap_class):
            maps[c] = ap[i, 0]
    else:
        p, r, mp, mr, map50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        print_log('none labels', logger)
        maps = np.zeros(nc) + map


    # Print speeds
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image

    shape = (batch_size, ch_in, imgsz[0], imgsz[1])
    print_log(f'all image number is {seen}, which cost {sum(dt)}s\nSpeed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t, logger)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))

    # Save JSON
    if save_json and len(jdict):
        det_coco_calculate(weights, data, task, save_dir, logger, jdict, last_conf, visual_matched)

    print_log(f"Results saved to {save_dir}", logger)


    det_result = (mp, mr, map50, map, (loss.cpu() / len(dataloader)).tolist())
    return det_result, maps, t

@torch.no_grad()
def predict(data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        augment=False,  # augmented inference
        half=True,  # use FP16 half-precision inference
        model=None,
        dataloader=None,
        save_dir=Path(''),
        bgr=1,
        last_conf=0.1,
        loss_num=3,
        visual_matched=False,
        logger=None,
        iof_nms=True,
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
    seen = 0

    names =  {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else [])}
    for ci, cls in enumerate(data.get('names', [])):
        names[ci] = cls

    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0


    pbar = tqdm(dataloader, desc='predict detection', bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar

    for batch_i, (im, targets, paths) in enumerate(pbar):

        # if batch_i >1:
        #     break
        seen += len(im)
        t1 = time_sync()
        im = im.to(device, non_blocking=True)


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

        out, train_out =  head_out[model.out_det_from]  # detect out

        dt[1] += time_sync() - t2
        # Loss


        if loss_num > 1:
            # NMS

            t3 = time_sync()
            out = non_max_suppression_with_iof(out, conf_thres, iou_thres, labels=[], multi_label=True, iof_nms=iof_nms)
            dt[2] += time_sync() - t3

            # Metrics
            for si, pred in enumerate(out):
                path = Path(paths[si])

                if visual_matched:
                    pred_b_s_l = pred[:, :6].cpu().numpy()  # （300，6）

                    visual_match_pred(im[si], None, pred_b_s_l, None, path, save_dir=save_dir, names=names,
                                      conf_ts=last_conf) #correct is shape of [n, div_area=4, iou=10]

        elif loss_num == 1:

            if visual_matched:
                save_object(im ,None, out.cpu(), paths, save_dir=save_dir, conf_ts=last_conf, f_map=train_out)


        if model.out_det_from == 1:
            out, _ = head_out[0]  # detect out
            if visual_matched:
                save_object(im ,None, out.cpu(), paths, save_dir=save_dir, conf_ts=(getattr(model, 'class_conf', 0.4)))

    dt[2] += 0.
    # Print speeds
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    shape = (batch_size, ch_in, imgsz[0], imgsz[1])
    print_log(
        f'all image number is {seen}, which cost {sum(dt)}s\nSpeed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t, logger)
