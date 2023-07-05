from utils import intersect_dicts, de_parallel
from copy import deepcopy
from models import Model
import torch
from glob import glob
from pathlib import Path
import sys


if __name__ == '__main__':
    root_dir = Path(__file__).resolve().parents[2] #'/home/workspace'
    # sys.path.append(f"{root_dir}/yolov5")
    # yolov5_path = str(root_dir/'yolov5')
    # sys.path.insert(0, yolov5_path)

    statedict_dir = 'zjdet/checkpoints/yolov5/'
    # statedict_dir = 'yolov5/state_dict/'
    save_str = 'zjdet/checkpoints'
    device = 'cuda:0'
    yamlf = 'configs/model/yolov5s.yaml'
    for weight in glob(f'{root_dir}/{statedict_dir}/*.pt'):
        file_name = Path(weight).name  # name stem parents
        ckpt = torch.load(weight)  # load checkpoint
        csd = ckpt # ["state_dict"]
        model = Model(yamlf, ch=3, nc=4, anchors=None).to(device)  # create
        csd = intersect_dicts(csd, model.state_dict(), exclude=[], key_match=False)  # intersect
        # print(csd)
        model.load_state_dict(csd, strict=False)
        model.hyp = ckpt.get('hyp', None)
        new_cpkt = {'epoch': ckpt.get('epoch',-1),
                    'model': deepcopy(de_parallel(model)).half(),
                    'optimizer': ckpt.get('optimizer',None)}
        torch.save(new_cpkt, Path(f'{root_dir}/{save_str}/zjs_{file_name}'))