from utils import intersect_dicts, de_parallel
from copy import deepcopy
from models import Model
import torch
from pathlib import Path
import numpy as np


if __name__ == '__main__':
    root_dir = Path(__file__).resolve().parents[1] #'/home/workspace/zjdet'
    save_str = f'{root_dir}/checkpoints'
    device = 'cuda:0'
    yamlf = f'{root_dir}/configs/model/zjdet_cat_filter_detect.yaml'
    weight = f'{root_dir}/checkpoints/zjdet_neck_small_airplane_3_7G_84_85_over25_84_92.pt'
    cls_weight = f'{root_dir}/results/train/merge_scls/c3_classify_s/exp3/weights/best_precision.pt'
    file_name = Path(yamlf).stem  # name stem parents
    ckpt = torch.load(weight)  # load checkpoint
    csd = ckpt['ema' if ckpt.get('ema') else 'model'].state_dict()
    model = Model(yamlf, ch=1, nc=1, anchors=None).to(device)  # create
    back_from = 9

    new_csd = {}
    for k, v in csd.items():
        strs = k.split('.')
        if int(strs[1])>7:
            back_from = 10 # because of cat in backbone of 8
        strs[1] = str(int(strs[1])+back_from)
        new_k = '.'.join(strs)
        if new_k in model.state_dict():
            mod_shape = model.state_dict()[new_k].shape #  ch_out, ch_in, _, _
            if v.shape == mod_shape:
                new_csd[new_k] = v
            else:
                # from functools import reduce
                # new_csd[new_k] = v.flatten()[-reduce(lambda x,y:x*y,mod_shape):].reshape(mod_shape)
                new_csd[new_k] = v.flatten()[-np.prod(mod_shape):].reshape(mod_shape)

    cls_ckpt = torch.load(cls_weight)  # load checkpoint
    cls_csd = cls_ckpt['ema' if cls_ckpt.get('ema') else 'model'].state_dict()

    # print(csd)
    model.load_state_dict(cls_csd, strict=False)
    model.load_state_dict(new_csd, strict=False)

    model.hyp = ckpt['hyp']if ckpt.get('hyp') else ckpt['model'].hyp
    new_cpkt = {'epoch': ckpt['epoch'],
                'model': deepcopy(de_parallel(model)).half(),
                'optimizer': ckpt['optimizer']
                }
    torch.save(new_cpkt, Path(f'{save_str}/{file_name}.pt'))