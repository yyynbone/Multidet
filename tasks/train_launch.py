import argparse
import os
import random
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch.distributed as dist
import yaml
from copy import deepcopy
import torch.multiprocessing as mp
from multiprocessing.shared_memory import ShareableList
from loss import *
from utils import (check_file, check_yaml, colorstr, get_latest_run, increment_path, print_args, print_mutation, load_args,
                   set_logging,  fitness, plot_evolve, set_cuda_visible_device, torch_distributed_zero_first)
from tasks import train, before_train

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # zjdet root directory

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'checkpoints/zjs_s16.pt', help='initial weights path')
    parser.add_argument('--opt-file', type=str, default='', help='opt file which load')
    parser.add_argument('--cfg', type=str, default=ROOT/'configs/model/zjdet_s16.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'configs/data/dota_eo_merged.yaml', help='dataset.yaml path')
    parser.add_argument('--data-prefile', type=str, default=ROOT / 'configs/preprocess/data_preprocess.yaml',
                        help='data preprocess.yaml path')
    parser.add_argument('--teacher', type=str, default='', help='teacher weight')

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
    parser.add_argument('--workers', type=int, default=4, help='max dataloader workers (per RANK in DDP mode)')
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
    parser.add_argument('--shuffle-epoch', type=int, default=1000, help='shuffle crop dataset')

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
    load_args(opt)
    return opt


def ready(opt):
    if opt.project == '':
        data_file = Path(opt.data).stem
        cfg_file = Path(opt.cfg).stem
        opt.project = ROOT / 'results/train' / str(data_file) / str(cfg_file)

        # Resume
    if opt.resume and not opt.evolve:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        with open(Path(ckpt).parent.parent / 'opt.yaml', errors='ignore') as f:
            opt = argparse.Namespace(**yaml.safe_load(f))  # replace
        opt.weights, opt.resume =  ckpt, True  # reinstate

    else:
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
            check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)  # checks
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        if opt.evolve:
            opt.project = str(ROOT / 'results/evolve')
            opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

        # print("now we in rank", local_rank) # runs in each rank
    opt.data_prefile = check_yaml(opt.data_prefile)
    Path(opt.save_dir).mkdir(parents=True, exist_ok=True)
    logger = set_logging(name=FILE.stem, filename=Path(Path(opt.save_dir) / 'train.log'))
    print_log(f"{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')} Now we start training in  {logger}", logger)
    print_args(FILE.stem, opt, logger=logger)
    # train_dataset, val_dataset = before_train(opt.hyp, opt, logger)
    # return opt, train_dataset, val_dataset
    opt, train_data, train_preprocess,  val_data, val_preprocess = before_train(opt.hyp, opt, logger)
    return opt, train_data, train_preprocess, val_data, val_preprocess

def main(local_rank, local_size, opt, train_data, train_preprocess, val_data, val_preprocess):
    # print('now we are in local_rank', local_rank)
    # local_rank could be 0-7
    addr, port, node, world_size = opt.addr, opt.port, opt.node, opt.world_size
    # set logger save to file:
    # Checks
    # print(f'in {local_rank}, save_dir is {opt.save_dir}, exist_ok is {opt.exist_ok}, {opt.name}')
    # DDP mode

    if node != -1:
        assert torch.cuda.device_count() > local_rank, 'insufficient CUDA devices for DDP command'
        # assert opt.batch_size % world_size == 0, '--batch-size must be multiple of CUDA device count'
        # assert not opt.image_weights, '--image-weights argument is not compatible with DDP training'
        assert not opt.evolve, '--evolve argument is not compatible with DDP training'
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        Rank = local_rank+node*local_size
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo",
                                init_method="tcp://{}:{}".format(addr, port),
                                rank=Rank,
                                world_size=world_size)
    else:
        opt.device = str(opt.device).strip().lower().replace('cuda:', '')  # to string, 'cuda:0' to '0'
        if opt.device == 'cpu':
            device = 'cpu'
        elif opt.device == '':
            device = 'cuda:0'
        else:
            device = f'cuda:{opt.device}'
        device = torch.device(device)
        # device = select_device(opt.device, batch_size=opt.batch_size, rank=local_rank)
    with torch_distributed_zero_first(local_rank):   # multiprocess all start
        logger = set_logging(name=FILE.stem, filename=Path(Path(opt.save_dir) / 'train.log'), rank=local_rank)  # 需要重新定义
            # logger.setLevel(20) # 20means logging.INFO, 但此时handle全部清空，即无file handle

    # Train
    if not opt.evolve:
        _ = train(opt, logger, device, train_data, train_preprocess, val_data, val_preprocess, local_rank, node, world_size)
        if world_size > 1 and node == 0:
            print_log('Destroying process group... ', logger)
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
            results = train(hyp.copy(), opt, logger, device, local_rank, node, world_size)
            # Write mutation results
            print_mutation(results, hyp.copy(), save_dir)

        # Plot results
        plot_evolve(evolve_csv)
        print_log(f'Hyperparameter evolution finished\n'
                    f"Results saved to {colorstr('bold', save_dir)}\n"
                    f'Use best hyperparameters example: $ python train.py --hyp {evolve_yaml}', logger)


def run(**kwargs):
    # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)


if __name__ == "__main__":
    opt = parse_opt()
    # main(opt)
    devices = set_cuda_visible_device(opt.device)
    # print(os.environ['CUDA_VISIBLE_DEVICES'])
    gpus = len(devices)
    if opt.world_size != gpus:
        opt.world_size = gpus #gpus
    
    opt, train_data, train_preprocess, val_data, val_preprocess = ready(opt)
    local_rank = opt.local_rank
    main(local_rank, gpus, opt, train_data, train_preprocess, val_data, val_preprocess)


#python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT  $(dirname "$0")/train.py $CONFIG  ${@:3}

# python -m torch.distributed.launch --nproc_per_node=8  tasks/train_launch.py --opt-file opt_zjdetunet_drone.yaml