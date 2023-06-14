# Ultralytics YOLO 🚀, GPL-3.0 license
"""
Common modules
"""

import time
import torch
import torch.nn as nn
import numpy as np
import math
from utils import Annotator

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(1, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        # x_short = self.cv2(x)
        # x_main = self.cv1(x)
        # x_main = self.m(x_main)
        #
        # x_final = torch.cat((x_main, x_short), dim=1)
        # return self.cv3(x_final)
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)

class Detect(nn.Module):
    stride = None  # strides computed during build

    def __init__(self, ch=(), nc=80, anchors=(),  inplace=True):  # detection layer
        super().__init__()
        #anchor [[10,13,16,30,33,23],...] shape(3,6)
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape   # xi(bs,255,20,20) to xi(bs,3,20,20,85)

            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0,1, 3, 4, 2).contiguous()

            if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

            y = x[i].sigmoid()

            y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

            z.append(y.view(bs, -1, self.no))

        return torch.cat(z, 1)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)], indexing='ij')
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid

    def bias_init(self,cf=None): # initialize biases into Detect(), cf is class frequency

        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        anchors = [[8, 15, 18, 30, 25, 15], [32, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
        self.layers = [Conv(3, 32, k=6, s=2, p=2),  # 0-p1/2
                       Conv(32, 64, k=3, s=2),  #1-p2/4
                       C3(64, 64, n=2),
                       Conv(64, 128, k=3, s=2),  # 3-p3/8
                       C3(128, 128, n=4),
                       Conv(128, 256, k=3, s=2),  #5-p4/16
                       C3(256, 256, n=6),
                       Conv(256, 512, k=3, s=2),  #7-p5/32
                       C3(512, 512, n=2),
                       SPPF(512, 512, k=5),  #9

                       Conv(512, 512, k=1, s=1),
                       C3(512, 512, n=2,  shortcut=False),
                       Conv(512, 256, k=1, s=1),
                       nn.Upsample(None, 2, 'nearest'),
                       Concat(),  # cat [-1,6], backbone p4
                       C3(512, 256, n=2, shortcut=False),  #12
                       Conv(256, 128, k=1, s=1),
                       nn.Upsample(None, 2, 'nearest'),
                       Concat(),  # cat [-1,4], backbone p3
                       C3(256, 128, n=2, shortcut=False),

                       Detect(nc=4, ch=(128, 256, 512), anchors=anchors),
                       ]
        self.cat_index = [(-1,6), (-1,4),(19,15,11)]
        self.save_index = set([ind for inds in self.cat_index for ind in inds if ind!=-1])
        self.save_x = [0 for _ in range(len(self.layers))]
        self.model = nn.Sequential(*self.layers)
        self.stride = self.model[-1].stride = torch.Tensor([8,16,32])
        self.nc = self.model[-1].nc

    def forward(self, x, **kargs):
        cat_i = 0
        for i, m in enumerate(self.model):
            if isinstance(m, (Concat, Detect)):
                x = [ x if ind==-1 else self.save_x[ind] for ind in self.cat_index[cat_i] ]
                x = m(x)
                cat_i += 1
            else:
                x = m(x)
            if i in self.save_index:
                self.save_x[i] = x
        return x

def convert_model():
    from utils import intersect_dicts
    device = 'cpu'
    weight = '/home/zjlab/Workspace/Hzh/zjdet/results/train/228/zjdet_neck/exp/weights/best_precision.pt'
    model = Model().to(device)
    ckpt = torch.load(weight, device)
    csd = ckpt['model'].float().state_dict()
    print(len(csd))
    csd = intersect_dicts(csd, model.state_dict(), exclude=[], key_match=False)  # intersect
    # print(csd)
    model.load_state_dict(csd, strict=False)
    print(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weight}')
    torch.save(model, '../convert_model.pt')

def convert_statedict():
    from utils import intersect_dicts
    device = 'cpu'
    weight = '../results/train/drone/zjdet_neck/exp2/weights/best.pt'
    model = Model().to(device)
    ckpt = torch.load(weight, device)
    csd = ckpt['model'].float().state_dict()
    print(len(csd))
    csd = intersect_dicts(csd, model.state_dict(), exclude=[], key_match=False)  # intersect
    # print(csd)
    model.load_state_dict(csd, strict=False)
    print(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weight}')
    torch.save(model.state_dict(), '../drone_model_statedict.pt', _use_new_zipfile_serialization=False)
def draw_box(im, box, label, color=(0, 0, 255), txt_color=(255, 255, 255)):
    lw = max(round(sum(im.shape) / 2 * 0.003), 2)
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(im, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h - 3 >= 0  # label fits outside box
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(im, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, lw / 3, txt_color,
                    thickness=tf, lineType=cv2.LINE_AA)
    return  im

if __name__ == '__main__':
    device = 'cpu'

    # convert_model()
    # convert_statedict()
    weight = '../drone_model_statedict.pt'
    state_dict = torch.load(weight, device)
    model = Model()
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    im = torch.zeros(1, 3, 540, 960)
    img_path = '../results/val/drone/zjdet_neck_exp2_best/exp/images/0000007_04999_d_0000036-400_1360_225_765.jpg'
    bgr = True
    import cv2
    if not bgr:
        img = cv2.imread(img_path, 0)
        im[0] = torch.tensor(img[:, :][None]/255.)
    else:
        img = cv2.imread(img_path)
        im[0] = torch.tensor(img.transpose(2,0,1)/ 255.)
    start = time.time()

    pred = model(im)
    print(pred.size())
    from utils import non_max_suppression
    pred = non_max_suppression(pred, 0.4, 0.7,  None,  False, 1000)
    # last conf filter
    # pred = [det[det[..., 4] >= 0.5] for det in pred]
    print(pred)
    end = time.time()
    print("####################")
    print(f'inference cost per image of {im.shape} %.2f ms'%((end-start)/20/16*1E3))
    print('done')
    names = [ 'car', 'van', 'truck', 'bus']
    # plot
    for i, det in enumerate(pred):  # per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            # det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class

            # Write results
            for *xyxy, conf, cls in reversed(det):
                # if save_txt:  # Write to file
                #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                #     line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                #     with open(txt_path + '.txt', 'a') as f:
                #         f.write(('%g ' * len(line)).rstrip() % line + '\n')


                c = int(cls)  # integer class
                label = (f'{names[c]} {conf:.1f}')
                img = draw_box(img, xyxy,label)


        cv2.imshow('result', img)
        cv2.waitKey(0)  # 1 millisecond


