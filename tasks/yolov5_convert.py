from utils import intersect_dicts, de_parallel
from copy import deepcopy
from models import Model
import torch
from glob import glob
from pathlib import Path
import sys


if __name__ == '__main__':
    root_dir = Path(__file__).resolve().parents[2] #'/home/workspace'
    statedict_dir = 'yolov5/state_dict'
    save_str = 'zjdet/checkpoints'
    device = 'cuda:0'
    yamlf = 'configs/model/zjdet_s16.yaml'
    for weight in glob(f'{root_dir}/{statedict_dir}/*.pt'):
        file_name = Path(weight).name  # name stem parents
        ckpt = torch.load(weight)  # load checkpoint
        csd = ckpt["state_dict"]
        model = Model(yamlf, ch=1, nc=80, anchors=None).to(device)  # create
        csd = intersect_dicts(csd, model.state_dict(), exclude=[])  # intersect
        # print(csd)
        model.load_state_dict(csd, strict=False)
        model.hyp = ckpt['hyp']
        new_cpkt = {'epoch': ckpt['epoch'],
                    'model': deepcopy(de_parallel(model)).half(),
                    'optimizer': ckpt['optimizer']}
        torch.save(new_cpkt, Path(f'{root_dir}/{save_str}/zjs_{file_name}'))