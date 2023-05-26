from utils import intersect_dicts, de_parallel
from copy import deepcopy
from models import Model
import torch
from glob import glob
from pathlib import Path
import sys


if __name__ == '__main__':
    # root_dir = Path(__file__).resolve().parents[2] #'/home/workspace'
    weight = 'checkpoints/squeezenet_statedict.pt'
    save_str = 'checkpoints'
    device = 'cpu'
    yamlf = 'configs/model/squeezenet.yaml'

    file_name = Path(weight).name  # name stem parents
    ckpt = torch.load(weight)  # load checkpoint
    csd = ckpt['state_dict']
    model = Model(yamlf, ch=1, nc=2, anchors=None).to(device)  # create
    model_sd = model.state_dict()
    csd = intersect_dicts(csd, model_sd, exclude=['backbone.final_conv'], key_match=False, skip=True)  # intersect
    # print(csd)
    model.load_state_dict(csd, strict=False)

    new_cpkt = {'model': model}
    torch.save(new_cpkt, Path(f'{save_str}/zjs_{file_name}'))