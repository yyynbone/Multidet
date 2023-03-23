import argparse
import math
import os
import random
from glob import glob
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader, distributed

import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD, Adam, lr_scheduler
from tqdm import tqdm
import torch.multiprocessing as mp

from models  import (attempt_load, Model)
from dataloader import check_anchors, check_train_batch_size, LoadImagesAndLabels, InfiniteDataLoader
from loss import *
from utils import (check_dataset, check_file, check_img_size, check_suffix, check_yaml, colorstr, cal_flops,
                   get_latest_run, increment_path, init_seeds, intersect_dicts, labels_to_class_weights,
                   labels_to_image_weights, one_cycle, linear, soft_cos_log, print_args, print_mutation, strip_optimizer,
                   set_logging, Callback, fitness, plot_evolve, plot_results, EarlyStopping, ModelEMA,
                   de_parallel, select_device, torch_distributed_zero_first, LossSaddle, select_class_tuple)
from tasks import val

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # zjdet root directory


def train(hyp, opt, logger, device, Local_rank=-1, Node=-1, World_size=1):
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze, = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze

    # callback only in local_rank = 0
    if Node in [-1,0] and Local_rank in [-1,0]:
        callbacks = Callback(save_dir, opt, opt.hyp, logger)  # callback instance
    # else:
    #     callbacks = Callback(save_dir, opt, opt.hyp, logger, include=())  # callback instance without tb
    # Directories
    w = save_dir / 'weights'  # weights dir
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'
    best_pt, best_rt = w / 'best_precision.pt', w / 'best_recall.pt'

    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict

    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    # Save run settings
    if not evolve:
        with open(save_dir / 'hyp.yaml', 'w') as f:
            yaml.safe_dump(hyp, f, sort_keys=False)
        with open(save_dir / 'opt.yaml', 'w') as f:
            yaml.safe_dump(vars(opt), f, sort_keys=False)

    # Loggers
    data_dict = None

    # Config
    plots = not evolve  # create plots
    cuda = device.type != 'cpu'
    init_seeds(1 + Node)
    with torch_distributed_zero_first(Local_rank):
        # this will run in each local_rank
        data_dict = data_dict or check_dataset(data)  # check if None
    train_path, val_path = data_dict['train'], data_dict['val']
    nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    # assert len(names) == nc, f'{len(names)} names found for nc={nc} dataset in {data}'  # check

    # Model
    check_suffix(weights, '.pt')  # check weights
    pretrained = weights.endswith('.pt')
    if opt.bgr:
        ch_in = 3
    else:
        ch_in = 1
    if pretrained:

        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        model = Model(cfg or ckpt['model'].yaml, ch=ch_in, nc=nc, anchors=hyp.get('anchors'), logger=logger).to(device)  # create

        # exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else active_layer # exclude keys
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys

        # csd =  ckpt['ema'].float().state_dict() if 'ema' in ckpt.keys()  else ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        if ckpt.get('ema'):
            csd = ckpt['ema'].float().state_dict()
            csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        elif ckpt.get('model'):
            csd = ckpt['model'].float().state_dict()
            csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        else:
            csd = ckpt['state_dict']
            csd = intersect_dicts(csd, model.state_dict(), exclude=exclude, key_match=False)  # intersect
        model.load_state_dict(csd, strict=False)  # load
        logger.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report

    else:
        model = Model(cfg, ch=ch_in, nc=nc, anchors=hyp.get('anchors'), logger=logger).to(device)  # create

    # Freeze
    task_dict = {1: 'encoder', 2: 'neck', 3: 'head'}
    freeze = list(range(freeze))

    neck_from_idx = model.neck_from
    head_from_idx = model.head_from

    para_idx = [[i for i in range(0, neck_from_idx)],
                [i for i in range(neck_from_idx, head_from_idx)],
                [i for i in range(head_from_idx, len(model.model)+1)],
                ]

    if not opt.task:  # [encoder,det,da_seg,l1_seg]
        pass
    else:
        for i in range(1, len(para_idx)):
            if not i in opt.task:
                logger.info(f'freeze {task_dict[i]}')
                freeze += para_idx[i - 1]
    # active_layer = [f'model.{x}.' for x in range(len(model.layer)) if x not in freeze]
    # freeze = [f'model.{x}.' for x in range(freeze)]  # layers to freeze

    freeze = [f'model.{x}.' for x in set(freeze)]

    model.freeze = freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            logger.info(f'freezing {k}')
            v.requires_grad = False

    # with torch_distributed_zero_first(Local_rank):
    #     show_model_param(model,'origin')

    # set model.train_val_filter
    model.train_val_filter = opt.train_val_filter

    # Image size
    gs = int(model.stride.max()) if hasattr(model, 'stride') else 32
    gs = max(gs, 32)  # grid size (max stride)
    imgsz = [opt.imgsz[0], opt.imgsz[0]] if len(opt.imgsz)==1 else opt.imgsz
    imgsz = check_img_size(imgsz, gs, floor=gs * 2, logger=logger)  # verify imgsz is gs-multiple
    # if isinstance(imgsz, int):
    #     imgsz = (imgsz, imgsz)
    # else:
    #     imgsz = tuple(imgsz)  # w,h

    # Batch size
    if Node == -1 and batch_size == -1:  # single-GPU only, estimate best batch size
        batch_size = check_train_batch_size(model, imgsz)

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

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
    logger.info(f"{colorstr('optimizer:')} {type(optimizer).__name__} with parameter groups "
                f"{len(g0)} weight, {len(g1)} weight (no decay), {len(g2)} bias")
    del g0, g1, g2

    # Scheduler
    if opt.linear_lr:
        lf = linear(1, hyp['lrf'], epochs)  # linear

    elif hyp['lrf']<0.1 and epochs>1000:
        lf = soft_cos_log(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA
    ema = ModelEMA(model) if Node in [-1,0] and Local_rank in [-1,0] else None

    # Resume
    start_epoch, best_fitness, best_p , best_r= 0, 0.0, 0.0, 0.0
    if pretrained and len(csd) == len(model.state_dict()):
        # Optimizer
        if ckpt['optimizer'] is not None and resume:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']
            logger.info('now load optimizer param done')

        # EMA
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']
            logger.info('now load ema state dict param done')

        # Epochs
        # start_epoch = 0
        if resume:
            logger.info(f'Resuming training from {opt.weights}')
            ckpt['epoch'] = ckpt.get('epoch', -1)
            start_epoch = ckpt['epoch'] + 1
            assert start_epoch > 0, f'{weights} training to {epochs} epochs is finished, nothing to resume.'
        if epochs < start_epoch:
            logger.info(f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} more epochs.")
            epochs += ckpt['epoch']  # finetune additional epochs
        del ckpt, csd

    # DP mode
    if cuda and Node == -1 and torch.cuda.device_count() > 1:
        logger.warning('WARNING: DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n')
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  modify
        #model = BalancedDataParallel(32,model,dim =0).to(device)
        model = torch.nn.DataParallel(model)
        #  modify  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#


    # SyncBatchNorm
    if opt.sync_bn and cuda and Node != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info('Using SyncBatchNorm()')

    # Trainloader
    batch_size = batch_size // World_size
    train_dataset = LoadImagesAndLabels(train_path,
                                        img_size=imgsz,
                                        batch_size=batch_size,
                                        logger=logger,
                                        single_cls=single_cls,
                                        hyp=hyp,  # hyperparameters
                                        augment=True,
                                        rect=opt.rect,
                                        stride=gs,
                                        pad=.0,
                                        prefix=colorstr('train: '),
                                        is_bgr=opt.bgr,
                                        filter_str=opt.filter_str,
                                        filter_bkg=opt.ignore_bkg,
                                        select_class=select_class_tuple(data_dict))
    # calculate class_weight
    class_weights = labels_to_class_weights(train_dataset.labels, nc)

    label_pos_weight = 0
    if hyp['CE']:
        hyp['cls_weight'] = class_weights


    # if opt.image_weights:
    #     iw = labels_to_image_weights(train_dataset.labels, nc=nc, class_weights=class_weights / nc)  # image weights
    #     # eg:pic num is 10 , class_weights is (0.2,0.3,1,1,1,1,...)(because real class_num=2 (but to 80)
    #     # dataset.indices before is (0,1,2,...9)
    #     # but now is (2,3,2,3 ,4,2,0,0,3,2) like this, so some pic is not trained
    #     # dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)\
    #     # so modified to this:
    #     add_indices = random.choices(range(train_dataset.n), weights=iw, k=train_dataset.n)  # rand weighted idx
    #     train_dataset.indices = np.concatenate((train_dataset.indices, add_indices))

    batch_size = min(batch_size, len(train_dataset))
    if opt.rect:
        logger.warning('WARNING: --rect is incompatible with DataLoader shuffle, setting shuffle=False')
        shuffle = False
    else:
        shuffle = True
    sampler = None if Local_rank == -1 else distributed.DistributedSampler(train_dataset, shuffle=shuffle)
    loader = DataLoader if opt.image_weights else InfiniteDataLoader  # only DataLoader allows for attribute updates
    train_loader = loader(train_dataset,
                    batch_size=batch_size,
                    shuffle=shuffle and sampler is None,
                    num_workers=4,
                    sampler=sampler,
                    pin_memory=True,
                    collate_fn=LoadImagesAndLabels.collate_fn4 if opt.quad else LoadImagesAndLabels.collate_fn)


    mlc = 0 # int(np.concatenate(train_dataset.labels, 0)[:, 0].max())  # max label class
    nb = len(train_loader)  # number of batches
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'

    # Process 0
    if Node in [-1,0] and Local_rank in [-1,0]:
        if not resume:
            # Anchors
            if not opt.noautoanchor:
                check_anchors(train_dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz, logger=logger)
            model.half().float()  # pre-reduce anchor precision

        # Valloader
        val_dataset = LoadImagesAndLabels(val_path,
                                          img_size=imgsz,
                                          batch_size=batch_size,
                                          logger=logger,
                                          single_cls=single_cls,
                                          hyp=hyp,  # hyperparameters
                                          augment=False,
                                          rect=opt.rect,
                                          stride=gs,
                                          pad=.0,
                                          prefix=colorstr('val: '),
                                          is_bgr=opt.bgr,
                                          filter_str=opt.filter_str,
                                          filter_bkg=opt.ignore_bkg,
                                          select_class=select_class_tuple(data_dict))
        val_loader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=4,
                                sampler=None,
                                pin_memory=True,
                                collate_fn=LoadImagesAndLabels.collate_fn)

    fs = cal_flops(de_parallel(model), imgsz)
    # DDP mode
    if cuda and Node != -1:
        try:
            model = DDP(model, device_ids=[Local_rank], output_device=Local_rank)
        except:
            model = DDP(model, device_ids=[Local_rank], output_device=Local_rank, find_unused_parameters=True)

    if hasattr(de_parallel(model), 'det_head_idx'):

        nl = de_parallel(model).model[de_parallel(model).det_head_idx].nl  # number of detection layers (to scale hyps)
        hyp['box'] *= 3 / nl  # scale to layers
        hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
        # hyp['obj'] *= (max(imgsz) / 640) ** 2 * 3 / nl  # scale to image size and layers
        hyp['label_smoothing'] = opt.label_smoothing

    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = torch.from_numpy(class_weights).to(device) * nc  # attach class weights
    model.names = names

    # Start training
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    stopper = EarlyStopping(patience=opt.patience, logger=logger)
    # lr_coeft = torch.tensor(1.0, device=device)
    stop = False
    broadcast_list = [stop, 1.] # stop, lr_coeft
    out_saddle = LossSaddle(patience=opt.loss_supervision, logger=logger)

    compute_loss = eval(opt.loss)(model,logger=logger)  # init loss class
    loss_num = compute_loss.loss_num
    pos_schedule = linear(hyp.get('lr_start_pos_weight', 0), hyp.get('max_pos_weight', 10), epochs)

    logger.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'now with size of {imgsz}, {fs:.2f} GFLOPS\n'
                f'Using {train_loader.num_workers * World_size} dataloader workers\n'
                f'Each epoch batch_size is {batch_size},iter is {nb}\n'
                f"Epoch train image shape is {train_dataset.__getitem__(0)[0].shape}, which you should input as (ch, y, x)"
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting training for {epochs} epochs...')

    if opt.val_first:
        start_epoch-=1


    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        # print('before epoch after broadcast is', broadcast_list, Local_rank)
        lr_coeft = broadcast_list[1]
        # print('before epoch after broadcast is', lr_coeft, Local_rank)
        if not opt.val_first:
            model.train()
            # Update image weights (optional, single-GPU only)
            if opt.image_weights:
                cw = class_weights * (1 - maps) ** 2 / nc  # class weights
                iw = labels_to_image_weights(train_dataset.labels, nc=nc, class_weights=cw)  # image weights
                train_dataset.indices = random.choices(range(train_dataset.n), weights=iw, k=train_dataset.n)  # rand weighted idx


            # Update mosaic border (optional)
            # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
            # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

            if Node != -1:
                train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(train_loader)

            # print(logger.handlers)
            # if Node in [-1,0] and Local_rank in [-1,0]:
            if len(logger.handlers):
                logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
                pbar = tqdm(pbar, total=nb, ncols=200, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
            optimizer.zero_grad()

            mloss = torch.zeros(loss_num, device=device)  # mean losses

            if hasattr(compute_loss, 'pos_weight'):
                # if compute_loss.pos_weight is None:
                #     ft_num = targets[0].shape[0] - targets[0].sum()
                #     label_pos_weight = torch.tensor([ft_num / max(targets[0].sum(), 1)])
                epoch_pos_w = pos_schedule(epoch)
                if epoch_pos_w > compute_loss.label_pos_weight:
                    compute_loss.pos_weight = torch.tensor([epoch_pos_w])
                logger.info(f'now epoch_pos_w is {epoch_pos_w}, ClassifyLoss label_pos_weight is {compute_loss.label_pos_weight}, ClassifyLoss pos_weight is {compute_loss.pos_weight}')

            for i, (imgs, targets, paths, shapes) in pbar:  # batch -------------------------------------------------------------
                # class_label, labels_out, seg_img = targets
                targets = [t.to(device)  if t is not None else t for t in targets]
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
                        x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

                # add lr grows up when in saddle point
                # 其实这是每个iter都会做这步，如有33个iter， x['lr']最后会变成 a*(lr_coeft**33),故设置每5个iter增加一次
                # if i%(nb//3) == 0: #每个epoch 4 次
                if i == 0:
                    for j, x in enumerate(optimizer.param_groups):
                        x['lr'] *= lr_coeft
                        if j==2 and ni<=nw:
                            x['lr'] = min(x['lr'], hyp['warmup_bias_lr'])
                        else:
                            x['lr'] = min(x['lr'], x['initial_lr'])

                # if lr_coeft >1. :
                #     logger.info(f"in epoch {epoch}, lr_coefficient is {lr_coeft}, lr is {x['lr']}")

                # Multi-scale
                if opt.multi_scale:
                    sz = random.randrange(max(imgsz) * 0.5, max(imgsz) * 1.5 + gs) // gs * gs  # size
                    sf = sz / max(imgs.shape[2:])  # scale factor
                    if sf != 1:
                        ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                        imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

                # Forward
                with amp.autocast(enabled=cuda):
                    head_out = model(imgs)  # forward
                    det_pred = head_out[de_parallel(model).out_det_from]

                    loss, loss_items = compute_loss(det_pred, targets)  # loss scaled by batch_size
                    if Node != -1:
                        loss *= World_size  # gradient averaged between devices in DDP mode
                    if opt.quad:
                        loss *= 4.

                # Backward

                scaler.scale(loss).backward()
                # Optimize
                # each iter batchsize must bigger then 64, eg：if bs is 128, accumulate=1 , if bs is 32, accumulate=2
                if ni - last_opt_step >= accumulate:
                    # clip the grad, to avoid the grad eruption(nan), clip_grad_norm_ or clip_grad_value_
                    scaler.unscale_(optimizer)  # Divides ("unscales") the optimizer's gradient tensors by the scale factor，这里统计并存入dict, optimizer_state["found_inf_per_device"]，check inf的数量，无则为0
                    # # calculate grad norm,so that we can set max_norm
                    # parameters = [p for p in model.parameters() if p.grad is not None]
                    # total_norm = torch.norm(
                    #     torch.stack([torch.norm(p.grad.detach(), 2).to(device) for p in parameters]), 2)
                    # print(total_norm)
                    # total_norm in (100,200)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20)
                    # print(f"now optimizer step, ni{ni} - last_opt_step{last_opt_step} >= accumulate{accumulate}, nbs is {nbs}, batch_size is {batch_size}")
                    scaler.step(optimizer)  # optimizer.step  # If inf/NaN gradients are found, `optimizer.step()` is skipped to avoid corrupting the params,使用self._maybe_opt_step，
                    # print("optimizer._step_count is ", optimizer._step_count) # 前5个iter ,一直为0，第6个开始为1，然后开始累加  #由上分析知，梯度含inf值
                    scaler.update()
                    optimizer.zero_grad()
                    if ema:
                        ema.update(model)
                        # if i % 20 == 19:
                        #     show_model_param(ema.ema,f'after_{epoch}_{i}')
                    last_opt_step = ni

                # change callbacks keys
                if i==0  and Node in [-1, 0] and Local_rank in [-1, 0]:
                    # if isinstance(loss, (tuple,list)):
                    if loss_num == 3:
                        logger.info(
                            ('\n' + '%10s' * 7) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'labels', 'img_size'))
                    elif loss_num == 4:
                        logger.info(
                            ('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'seg', 'labels', 'img_size'))
                        callbacks.keys = ['train/box_loss', 'train/obj_loss', 'train/cls_loss', 'train/seg_loss'  # train loss
                     'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',  # metrics
                     'val/box_loss', 'val/obj_loss', 'val/cls_loss',  'val/seg_loss'# val loss
                     'x/lr0', 'x/lr1', 'x/lr2']  # params
                    else:
                        callbacks.keys = ['train/cls_loss',  # train loss
                                          'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5',
                                          'metrics/mAP_0.5:0.95',  # metrics
                                          'val/cls_loss',  # val loss
                                          'x/lr0', 'x/lr1', 'x/lr2']  # params
                        logger.info(
                            ('\n' + '%10s' * 7) % ('Epoch', 'gpu_mem',  'cls', 'all_pic', 'obj', 'labels', 'img_size'))

                # Log
                if Node in [-1,0] and Local_rank in [-1,0]:
                    mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                    mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                    if loss_num == 1:
                        pbar.set_description(('%10s' * 2 + '%10.4g' * (loss_num+4)) % (
                            f'{epoch}/{epochs - 1}', mem, *mloss, targets[0].shape[0], targets[0].sum(), targets[1].shape[0], imgs.shape[-1]))
                    else:
                        pbar.set_description(('%10s' * 2 + '%10.4g' * (2+loss_num)) % (
                            f'{epoch}/{epochs - 1}', mem, *mloss, targets[1].shape[0], imgs.shape[-1]))

                    callbacks.on_train_batch_end(ni, model, imgs, targets, paths, plots, opt.sync_bn)

            # Scheduler
            lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
            # print("now scheduler step means lr_scheduler step")
            scheduler.step()
            if Node in [-1,0] and Local_rank in [-1,0] and opt.loss_supervision:
                lr_coeft = out_saddle(epoch=epoch, loss=loss.item())
                if lr_coeft>1:
                    logger.info(f'now in epoch {epoch}, lr coefficient is:{lr_coeft}')

        if Node in [-1,0] and Local_rank in [-1,0]:
            # mAP
            # callbacks.on_train_epoch_end(epoch=epoch)
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            if not noval or final_epoch:  # Calculate mAP

                if opt.val_train:
                    results, _, times = val.run(data_dict,
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
                    msg = 'Valuate traindataset Epoch: [{0}]    Loss({loss})\n' \
                          'Detect: P({p:.3f})  R({r:.3f})  mAP@0.5({map50:.3f})  mAP@0.5:0.95({map:.3f})\n' \
                          'Time: inference({t_inf:.4f}s/frame)  nms({t_nms:.4f}s/frame)'.format(
                        epoch, loss=loss_tuple,
                        p=results[0], r=results[1], map50=results[2], map=results[3],
                        t_inf=times[1], t_nms=times[2])
                    logger.info(msg)

                results, maps,times = val.run(data_dict,
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
                    logger.info(msg)

                    # fi = fitness(np.array(all_results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
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
                            torch.save(ckpt, w/'nanerupt.pt')
                            print("warning now wo found a nan in loss in val")
                            break
                        else:
                            torch.save(ckpt, last)
                        if fi > best_fitness:
                            best_fitness = fi
                            torch.save(ckpt, best)
                            logger.info(f'best epoch saved in Epoch {epoch}')

                        if recall_fi > 0.9 and precision_fi > 0.5 :
                            torch.save(ckpt, w / f'epoch{epoch}.pt')
                            logger.info(f'a better recall epoch saved in Epoch {epoch}')
                        if recall_fi>0.8:
                            re_fi = precision_fi + (1.5-0.8)/(1-0.8**2)*recall_fi
                            if re_fi > best_r :
                                best_r = re_fi
                                torch.save(ckpt, best_rt)
                                logger.info(f'best recall epoch saved in Epoch {epoch}')

                        pr_fi = 2.8 * precision_fi * recall_fi / np.maximum(1.8 * precision_fi + recall_fi, 1e-20)

                        if pr_fi > best_p and recall_fi > 0.6:
                            best_p = pr_fi
                            torch.save(ckpt, best_pt)
                            logger.info(f'best precision epoch saved in Epoch {epoch}')

                        if (epoch > 0) and (opt.save_period > 0) and (epoch % opt.save_period == 0):
                            torch.save(ckpt, w / f'epoch{epoch}.pt')
                        if (epoch < hyp['warmup_epochs']+ 3) and opt.save_warm :
                            torch.save(ckpt, w / f'epoch{epoch}.pt')
                        del ckpt
                        # callbacks.on_model_save(last, epoch, final_epoch, best_fitness, fi)

                    stop = stopper(epoch=epoch, fitness=fi)
                    plot_results(file=save_dir / 'results.csv')  # save results.png

        if opt.val_first:
            opt.val_first = False


        #####################stop and broadcast#######################
        # Stop Single-GPU
        if Node == -1:
            if stop:
                break
        else:
            # Stop DDP
            if Node == 0 and Local_rank == 0:
                broadcast_list = [stop, lr_coeft]
                # print('broadcast is', broadcast_list, Local_rank)
            # else:
            #     broadcast_list = [None, None]
                # print('broadcast is', broadcast_list, Local_rank)
            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
            # Stop DPP
            # print('now after broadcast is', broadcast_list, Local_rank)
            with torch_distributed_zero_first(Local_rank):
                stop = broadcast_list[0]
                if stop:
                   break  # must break all DDP ranks
            ############################################

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    if Node in [-1,0] and Local_rank in [-1,0]:
        logger.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        for f in glob(f'{w}/*.pt'):
            if os.path.exists(f):
                strip_optimizer(f)  # strip optimizers
                if f in (best, best_pt, best_rt):
                    logger.info(f'\nValidating {f}...')
                    results,  _, _  = val.run(data_dict,
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
        logger.info(f"Results saved to {colorstr('bold', save_dir)}")

    torch.cuda.empty_cache()
    return results


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'checkpoints/zjs_s16.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default=ROOT/'configs/model/zjdet_s16.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'configs/data/dota_eo_merged.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'configs/hyp/hyp.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--bgr', type=int, default=1,help='if 1 bgr,0 gray')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, nargs='+', default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default='', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=300, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', type=int, default=0, help='Number of layers to freeze. backbone=10, all=24')
    parser.add_argument('--task', type=int, nargs='+',default=0, help='task of 1:encoder,2:det,3:da_seg,4:l1_seg')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')

    parser.add_argument('--save-warm', type=int, default=0, help='Save checkpoint at the warm up period')
    parser.add_argument('--val-first', nargs='?', const=True, default=False, help='run val before epoch')
    parser.add_argument('--div-area', nargs='+', type=int, default=None, help='area of pixel to divide')
    parser.add_argument('--last-conf', type=float, default=0.1, help='last conf thresh after nms')
    parser.add_argument('--loss', type=str, default='LossV5', help='loss name used')
    parser.add_argument('--loss-supervision', type=int, default=0, help='loss epoch patiances of supervision')
    parser.add_argument('--filter-str', type=str, default='', help='filter and select the image name with string')
    parser.add_argument('--ignore-bkg', action='store_true', help='filter and image of background')
    parser.add_argument('--train-val-filter', action='store_true', help='filter first use the classify head')
    parser.add_argument('--val-train', action='store_true', help='valuate the train dataset')

    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--node', type=int, default=0, help='computer node')
    parser.add_argument('--addr', type=str, default='localhost', help='computer addr')
    parser.add_argument('--port', type=int, default=12345, help='computer addr')
    parser.add_argument('--world-size', type=int, default=4, help='computer gpu count')

    # Weights & Biases arguments
    parser.add_argument('--entity', default=None, help='W&B: Entity')
    parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='W&B: Upload data, "val" option')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='W&B: Set bounding-box image logging interval')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='W&B: Version of dataset artifact to use')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main(Local_rank, Local_size, opt):
    addr, port, Node, World_size = opt.addr, opt.port, opt.node, opt.world_size
    if opt.project == '':
        data_file = Path(opt.data).stem
        cfg_file = Path(opt.cfg).stem
        opt.project = ROOT / 'results/train' /str(data_file)/ str(cfg_file)

    # Resume
    if opt.resume  and not opt.evolve:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        with open(Path(ckpt).parent.parent / 'opt.yaml', errors='ignore') as f:
            opt = argparse.Namespace(**yaml.safe_load(f))  # replace
        opt.cfg, opt.weights, opt.resume = '', ckpt, True  # reinstate

    else:
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
            check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)  # checks
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        if opt.evolve:
            opt.project = str(ROOT / 'results/evolve')
            opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # set logger save to file:
    # Checks
    # print(f'in {Local_rank}, save_dir is {opt.save_dir}, exist_ok is {opt.exist_ok}, {opt.name}')
    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size, rank=Local_rank)
    if Node != -1:
        assert torch.cuda.device_count() > Local_rank, 'insufficient CUDA devices for DDP command'
        assert opt.batch_size % World_size == 0, '--batch-size must be multiple of CUDA device count'
        # assert not opt.image_weights, '--image-weights argument is not compatible with DDP training'
        assert not opt.evolve, '--evolve argument is not compatible with DDP training'
        torch.cuda.set_device(Local_rank)
        device = torch.device('cuda', Local_rank)
        Rank = Local_rank+Node*Local_size
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo",
                                init_method="tcp://{}:{}".format(addr, port),
                                rank=Rank,
                                world_size=World_size)


    with torch_distributed_zero_first(Local_rank):
        # print("now we in rank", Local_rank) # runs in each rank
        Path(opt.save_dir).mkdir(parents=True, exist_ok=True)
        logger = set_logging(name=FILE.stem, filename=Path(Path(opt.save_dir) / 'train.log'), rank=Local_rank)
        logger.info(f"Now we start training in{datetime.now().strftime('%Y-%m-%d_%H:%M:%S') }")
        print_args(FILE.stem, opt, logger=logger)

    # Train
    if not opt.evolve:
        _ = train(opt.hyp, opt, logger, device, Local_rank, Node, World_size)
        if World_size > 1 and Node == 0:
            logger.info('Destroying process group... ')
            dist.destroy_process_group()

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
                'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
                'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
                'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
                'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
                'box': (1, 0.02, 0.2),  # box loss gain
                'cls': (1, 0.2, 4.0),  # cls loss gain
                'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                'iou_t': (0, 0.1, 0.7),  # IoU training threshold
                'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
                'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
                'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
                'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
                'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
                'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
                'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
                'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
                'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
                'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
                'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
                'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
                'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
                'mixup': (1, 0.0, 1.0),  # image mixup (probability)
                'copy_paste': (1, 0.0, 1.0)}  # segment copy-paste (probability)

        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            if 'anchors' not in hyp:  # anchors commented in hyp.yaml
                hyp['anchors'] = 3
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # only val/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        evolve_yaml, evolve_csv = save_dir / 'hyp_evolve.yaml', save_dir / 'evolve.csv'

        for _ in range(opt.evolve):  # generations to evolve
            if evolve_csv.exists():  # if evolve.csv exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt(evolve_csv, ndmin=2, delimiter=',', skiprows=1)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min() + 1E-6  # weights (sum > 0)
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([meta[k][0] for k in hyp.keys()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation
            results = train(hyp.copy(), opt, logger, device, Local_rank, Node, World_size)
            # Write mutation results
            print_mutation(results, hyp.copy(), save_dir)

        # Plot results
        plot_evolve(evolve_csv)
        logger.info(f'Hyperparameter evolution finished\n'
                    f"Results saved to {colorstr('bold', save_dir)}\n"
                    f'Use best hyperparameters example: $ python train.py --hyp {evolve_yaml}')


def run(**kwargs):
    # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)


if __name__ == "__main__":
    opt = parse_opt()
    # main(opt)
    gpus = torch.cuda.device_count()
    opt.workers = int(opt.workers /4)
    opt.batch_size = int(opt.batch_size*opt.world_size)
    # print(opt.batch_size) 384
    if opt.world_size==1:
        opt.node = -1
        main(-1,gpus,opt)
    else:
        mp.spawn(main,
                 args=(gpus,opt),
                 nprocs=gpus,
                 join=True)


