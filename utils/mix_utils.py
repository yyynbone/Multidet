
import torch
import inspect
import glob

from pathlib import Path

import re
import datetime
import math
import os
import json
from utils.label_process import xyxy2xywh

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_default_args(func):
    # Get func() default arguments
    signature = inspect.signature(func)
    return {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}

def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')

def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 0, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({'image_name': image_id,
                      'category_id': class_map[int(p[5])],
                      'bbox': [round(x, 3) for x in b],
                      'score': round(p[4], 5)})

def show_model_param(model,file=None, path=None, layer_idx=None):
    data={}
    for i,(k, v) in enumerate(model.named_parameters()):
        # print(f"in layer {k}:  param value is:\n{v}")
        if layer_idx is not None:
            if isinstance(layer_idx, int):
                layer_idx = list(range(layer_idx))
            if i in layer_idx:
                data[k] = v.cpu().detach().numpy().tolist()
                print(f'in {k}, {v.shape}')
    if file is not None:
        with open(file, 'w') as f:
            # sys.stdout = f
            # sys.stderr = f
            json.dump(data, f, indent=4)
    if path is not None:
        path = increment_path(path, mkdir=True)
        with open(path/'para.json', 'w') as f:
            json.dump(data, f, indent=4)
        print("save",str(path))

def show_state_dict(state_dict,file=None, dict_idx=list(range(400, 410))):
    o_data={}
    for i, (k,v) in enumerate(state_dict.items()):
        if i in dict_idx:
            o_data[k] = v.cpu().detach().numpy().tolist()
            print(f'in {k}, {v.shape}')
    if file is not None:
        with open(file, 'w') as f:
            # sys.stdout = f
            # sys.stderr = f
            json.dump(o_data, f, indent=4)
def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d.split(os.sep)[-1]) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # increment path
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory
    return path

def date_modified(path=__file__):
    # return human-readable file modification date, i.e. '2021-3-26'
    t = datetime.datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f'{t.year}-{t.month}-{t.day}'

def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor

# def area_filter(array, e_class=4, filter_idx=1):
#     assert array.shape[0]%e_class==0, 'must be divided by e_class'
#     idx = []
#     for i in range(array.shape[0]):
#         if i%e_class !=filter_idx :
#             idx.append(i)
#     return array[idx]

def area_filter(array, filter_from=2):
    return array[:, filter_from:]

class ExpWA():
    def __init__(self,average=0., decay=0.999,iter=2000, updates=0):
        self.updates = updates  # number of EWA updates

        self.iter = iter
        #self.iter = 1/(1-decay)
        self.decay_f = lambda x: 1 - math.exp(-x /self.iter)
        self.decay = decay
        self.average = average
    def update(self,observe):
        self.updates+=1
        if self.updates < self.iter:
            d = ( 1-(self.updates/self.iter)/math.exp(1) )*self.decay

        else:
            d = self.decay_f(self.updates)*self.decay
        self.average = self.average*d + (1-d)*observe
        self.weight = self.average/observe

def smooth_BCE(eps=0.1):
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps

def intersect_dicts(da, db, exclude=(), key_match=True):
    if key_match:
        # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
        return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}
    else:
        new_csd = {}
        value = list(db.values())
        keys = list(db.keys())
        for i, (k, v) in enumerate(da.items()):
            if not any(x in k for x in exclude) and i<len(value):
                if v.shape == value[i].shape:
                    new_csd[keys[i]] = v
                else:
                    print(f'{k} shape {v.shape} not match {keys[i]} shape {value[i].shape}')
            else:
                print(f'{k} shape {v.shape} not match')
        return new_csd

def get_latest_run(search_dir='.'):
    # Return path to most recent 'last.pt' in /runs (i.e. to --resume from)
    last_list = glob.glob(f'{search_dir}/**/last*.pt', recursive=True)
    return max(last_list, key=os.path.getctime) if last_list else ''

def iter_extend_axis(x, filter_bool):
    if isinstance(x, list):
        for i,a in enumerate(x):
            x[i] = iter_extend_axis(a, filter_bool)
    elif isinstance(x, tuple):
        x = list(x)
        for i,a in enumerate(x):
            x[i] = iter_extend_axis(a, filter_bool)
        x = tuple(x)
    else:
        new_x = torch.zeros( (len(filter_bool), *x[0].shape), device=x.device, dtype=x.dtype) #, requires_grad=x.requires_grad )
        new_x[filter_bool] = x
        x = new_x
    return x

