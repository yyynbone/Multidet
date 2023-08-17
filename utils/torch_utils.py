import random
import numpy as np
import math
import time
import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from copy import deepcopy
import contextlib
from contextlib import contextmanager
from utils.mix_utils import date_modified
from utils.logger import print_log

def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    # cudnn seed 0 settings are slower and more reproducible, else faster and less reproducible
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)

def time_sync(device=torch.device('cuda:0')):
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    return time.time()

@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
         使用了 contextmanager 装饰器后，
         def  f():
             a = 1
             yield
             b = 2
         with f():
            c = 3

        这等价于：

        a = 1
        try:
            c = 3
        finally:
            n = 2

    """
    if local_rank not in [-1, 0]:
        dist.barrier(device_ids=[local_rank])
    yield # 保证yeild 后面的内容最后运行，即0 最后堵塞，故 rank=0 的程序先运行, 其它后运行
    if local_rank == 0:
        dist.barrier(device_ids=[0])

class Profile(contextlib.ContextDecorator):
    """
    YOLOv8 Profile class.
    Usage: as a decorator with @Profile() or as a context manager with 'with Profile():'
    """

    def __init__(self, t=0.0):
        """
        Initialize the Profile class.

        Args:
            t (float): Initial time. Defaults to 0.0.
        """
        self.t = t
        self.cuda = torch.cuda.is_available()

    def __enter__(self):
        """
        Start timing.
        """
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        """
        Stop timing.
        """
        self.dt = self.time() - self.start  # delta-time
        self.t += self.dt  # accumulate dt

    def time(self):
        """
        Get current time.
        """
        if self.cuda:
            torch.cuda.synchronize()
        return time.time()



def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model

def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True
def kaiming_normal(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform(m.weight.data)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
        if hasattr(m, 'bias'):
            if m.bias is not None:
                m.bias.data.zero_()

def fuse_conv_and_bn(conv, bn):
    # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))  #torch.diag 取矩阵对角线
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv

def nancheck(loss):
    if not isinstance(loss, torch.Tensor):
        loss = torch.tensor(loss)
    return torch.isnan(loss).sum()

def infcheck(loss):
    if not isinstance(loss, torch.Tensor):
        loss = torch.tensor(loss)
    return torch.isinf(loss).sum()

def set_cuda_visible_device(device='', newline=True, logger=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    s = f'ZJDetection {date_modified()} torch {torch.__version__} '  # string
    device = str(device).strip().lower().replace('cuda:', '')  # to string, 'cuda:0' to '0'
    cpu = device == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        devices = device.split(',') if device else range(torch.cuda.device_count())  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        space = ' ' * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2:.0f}MiB)\n"  # bytes to MB
    else:
        s += 'CPU\n'
        devices = []
    if not newline:
        s = s.rstrip()
    print_log(s, logger)
    return  devices

def select_device(device='', batch_size=None, newline=True, logger=None, rank=-1):
    # device = 'cpu' or '0' or '0,1,2,3'
    s = f'ZJDetection {date_modified()} torch {torch.__version__} '  # string
    device = str(device).strip().lower().replace('cuda:', '')  # to string, 'cuda:0' to '0'
    cpu = device == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        devices = device.split(',') if device else '0'  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size:  # check batch_size is divisible by device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2:.0f}MiB)\n"  # bytes to MB

        device = f'cuda:{devices[0]}'
    else:
        s += 'CPU\n'


    if not newline:
        s = s.rstrip()
    if rank in [-1.0]:
        print_log(s, logger)
    # return torch.device('cuda:0' if cuda else 'cpu')
    return torch.device(device)
    # return  devices

def cal_flops(model,img_size,verbose=False):
    from thop import profile
    stride = 256
    device = next(model.parameters()).device
    img = torch.zeros((1, model.yaml.get('ch_input', 3), stride, stride), device=device)  # input
    flops, params = profile(deepcopy(model), inputs=(img,), verbose=verbose)
    flops = flops / 1E9 * 2  #  GFLOPs
    # print(flops, params)
    img_size = img_size if isinstance(img_size, (list,tuple)) else [img_size, img_size]  # expand if int/float
    img_flops = flops * img_size[0] / stride * img_size[1] / stride # GFLOPs
    return img_flops

class EarlyStopping:
    # YOLOv5 simple early stopper
    def __init__(self, patience=30, logger=None):
        self.best_fitness = 0.0  # i.e. mAP
        self.best_epoch = 0
        self.patience = patience or float('inf')  # epochs to wait after fitness stops improving to stop
        self.possible_stop = False  # possible stop may occur next epoch
        self.logger=logger

    def __call__(self, epoch, fitness):
        if fitness >= self.best_fitness:  # >= 0 to allow for early zero-fitness stage of training
            self.best_epoch = epoch
            self.best_fitness = fitness
        delta = epoch - self.best_epoch  # epochs without improvement
        self.possible_stop = delta >= (self.patience - 1)  # possible stop may occur next epoch
        stop = delta >= self.patience  # stop training if patience exceeded
        if stop:
            print_log(f'Stopping training early as no improvement observed in last {self.patience} epochs. '
                        f'Best results observed at epoch {self.best_epoch}, best model saved as best.pt.\n'
                        f'To update EarlyStopping(patience={self.patience}) pass a new patience value, '
                        f'i.e. `python train.py --patience 300` or use `--patience 0` to disable EarlyStopping.', self.logger)
        return stop

class LossSaddle:
    # YOLOv5 simple out of saddle point
    def __init__(self, patience=50, logger=None):
        self.loss = 1e3
        self.best_epoch = 0
        self.patience = patience or float('inf')  # epochs to wait after fitness stops improving to stop
        self.coeft = 1.
        self.coeft_epoch = 0
        self.logger = logger

    def __call__(self, epoch, loss):
        if loss <= self.loss:  # >= 0 to allow for early zero-fitness stage of training
            self.best_epoch = epoch
            self.loss = loss
        
        if epoch > 4 * self.patience:
            delta = epoch - self.best_epoch  # epochs without improvement
            change_coeft = (delta >= self.patience) and (
                        epoch - self.coeft_epoch >= 2 * self.patience)  # stop training if patience exceeded
            if change_coeft:
                print_log(f'Loss isnt going down in last {self.patience} epochs. '
                            f'Best results observed at epoch {self.best_epoch}, now we rap out the saddle point.\n', self.logger)
                # #optimizer.paramgroups 只有当前epoch的参数，随着scheduler.step() 更新故不用考虑*0.5以乘回原样。
                # self.coeft = [0.5, .5, 1., 1., 1., 2., 2.]
                # self.coeft = [1., 10., 5., 2.]
                # self.coeft = [2., 3., 4., 5., 5. ]
                self.coeft_epoch = epoch
    
            # if isinstance(self.coeft, list):
                # if len(self.coeft):
                #     return self.coeft.pop()
                # else:
                #     self.coeft = 1.
            change_delta = epoch - self.coeft_epoch
            if change_delta >=0:
                if change_delta < self.patience//5:
                    self.coeft = min(max(10. * epoch //1000, 2), 10)
                else:
                    self.coeft /= 2.
        self.coeft = max(self.coeft, 1.)
        return self.coeft

def scale_img(img, ratio=1.0, same_shape=False, gs=32):  # img(16,3,256,416)
    # scales img(bs,3,y,x) by ratio constrained to gs-multiple
    if ratio == 1.0:
        return img
    else:
        h, w = img.shape[2:]
        s = (int(h * ratio), int(w * ratio))  # new size
        img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)  # resize
        if not same_shape:  # pad/crop img
            h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))
        return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean

def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)

class ModelEMA:
    def __init__(self, model, decay=0.9999, updates=0):
        # Create EMA
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            model_now = model.module if is_parallel(model) else model  # model
            msd = model_now.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point and (not any(x in k for x in model_now.freeze)) :
                    v *= d
                    v += (1 - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)

def strip_optimizer(f='best.pt', s=''):  # from utils.general import *; strip_optimizer()
    # Strip optimizer from 'f' to finalize training, optionally save as 's'
    x = torch.load(f, map_location=torch.device('cpu'))
    if x.get('ema'):
        x['model'] = x['ema']  # replace model with ema
    for k in 'optimizer', 'training_results', 'wandb_id', 'ema', 'updates':  # keys
        x[k] = None
    x['epoch'] = -1
    x['model'].half()  # to FP16
    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1E6  # filesize
    print(f"Set requires_grad to False, Optimizer stripped from {f},{(' saved as %s,' % s) if s else ''} {mb:.1f}MB")
