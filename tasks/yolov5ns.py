import torch
import torch.nn as nn
import warnings
import time
from thop import profile


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    """
    为same卷积或same池化作自动扩充（0填充）  Pad to 'same'
    :params k: 卷积核的kernel_size
    :return p: 自动计算的需要pad值（0填充）
    """
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution
    """
    Standard convolution  conv+BN+act
    :params c1: 输入的channel值
    :params c2: 输出的channel值
    :params k: 卷积的kernel_size
    :params s: 卷积的stride
    :params p: 卷积的padding  一般是None  可以通过autopad自行计算需要pad的padding数
    :params g: 卷积的groups数  =1就是普通的卷积  >1就是深度可分离卷积
    :params act: 激活函数类型   True就是SiLU()/Swish   False就是不使用激活函数
                 类型是nn.Module就使用传进来的激活函数类型
    """
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """
        前向融合计算  减少推理时间
        """
        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, c2o, n=1, shortcut=True, g=1, e=[0.5, 0.5], rate=[1.0 for _ in range(12)]):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        # c_ = int(c2 * e)  # hidden channels
        if isinstance(e, list):
            c1_ = int(c2o * e[0])
            c2_ = int(c2o * e[1])
        else:
            c1_ = int(c2o * e)
            c2_ = int(c2o * e)
        self.cv1 = Conv(c1, c1_, 1, 1)
        self.cv2 = Conv(c1, c2_, 1, 1)
        self.cv3 = Conv(c1_+c2_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c1_, c1_, shortcut, g, e=rate[i]) for i in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5, e=0.5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        # c_ = c1 // 2  # hidden channels
        c_ = int(c1*e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))

class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        # print(x[0].shape, x[1].shape)
        return torch.cat(x, self.d)

class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        # print(self.nl)
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
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid

class YOLOV5n(nn.Module):
    def __init__(self):
        super(YOLOV5n, self).__init__()

        self.conv0 = Conv(c1=1, c2=16, k=6, s=2, p=2)  # 0
        self.conv1 = Conv(c1=16, c2=32, k=3, s=2)  # 1
        self.c0 = C3(c1=32, c2=32, c2o=32, n=1)  # 2
        self.conv2 = Conv(c1=32, c2=64, k=3, s=2)  # 3
        self.c1 = C3(c1=64, c2=64, c2o=64, n=2)  # 4
        self.conv3 = Conv(c1=64, c2=128, k=3, s=2)  # 5
        self.c2 = C3(c1=128, c2=128, c2o=128, n=3)  # 6
        self.conv4 = Conv(c1=128, c2=256, k=3, s=2)  # 7
        self.c3 = C3(c1=256, c2=256, c2o=256, n=1)  # 8
        self.sppf0 = SPPF(c1=256, c2=256, k=5)  # 9
        self.conv5 = Conv(c1=256, c2=128, k=1, s=1)  # 10
        self.unsample0 = nn.Upsample(scale_factor=1.9, mode='nearest')  # 11
        self.concat0 = Concat()  # 12
        self.c4 = C3(c1=256, c2=128, c2o=128, n=1, shortcut=False)  # 13
        self.conv6 = Conv(c1=128, c2=64, k=1, s=1)  # 14
        self.unsample1 = nn.Upsample(scale_factor=2.0, mode='nearest')  # 15
        self.concat1 = Concat()  # 16
        self.c5 = C3(c1=128, c2=64, c2o=64, n=1, shortcut=False)  # 17
        self.conv7 = Conv(c1=64, c2=64, k=3, s=2)  # 18
        self.concat2 = Concat()  # 19
        self.c6 = C3(c1=128, c2=128, c2o=128, n=1, shortcut=False)  # 20
        self.conv8 = Conv(c1=128, c2=128, k=3, s=2)  # 21
        self.concat3 = Concat()  # 22
        self.c7 = C3(c1=256, c2=256, c2o=256, n=1, shortcut=False)  # 23
        # self.conv0 = Conv(c1=3, c2=16, k=6, s=2, p=2)  # 0
        # self.conv1 = Conv(c1=16, c2=32, k=3, s=2)  # 1
        # self.conv2 = Conv(c1=32, c2=64, k=3, s=2)  # 3
        # self.conv3 = Conv(c1=64, c2=128, k=3, s=2)  # 5
        # self.conv4 = Conv(c1=128, c2=256, k=3, s=2)  # 7
        # self.conv5 = Conv(c1=256, c2=128, k=1, s=1)  # 10
        # self.conv6 = Conv(c1=128, c2=64, k=1, s=1)  # 14
        # self.conv7 = Conv(c1=64, c2=64, k=3, s=2)  # 18
        # self.conv8 = Conv(c1=128, c2=128, k=3, s=2)  # 21

        # self.c0 = C3(c1=32, c2=32, c2o=32, n=1)  # 2
        # self.c1 = C3(c1=64, c2=64, c2o=64, n=2)  # 4
        # self.c2 = C3(c1=128, c2=128, c2o=128, n=3)  # 6
        # self.c3 = C3(c1=256, c2=256, c2o=256, n=1)  # 8
        # self.c4 = C3(c1=256, c2=128, c2o=128, n=1, shortcut=False)  # 13
        # self.c5 = C3(c1=128, c2=64, c2o=64, n=1, shortcut=False)  # 17
        # self.c6 = C3(c1=128, c2=128, c2o=128, n=1, shortcut=False)  # 20
        # self.c7 = C3(c1=256, c2=256, c2o=256, n=1, shortcut=False)  # 23

        # self.sppf0 = SPPF(c1=256, c2=256, k=5)  # 9
        #
        # self.unsample0 = nn.Upsample(scale_factor=2.0, mode='nearest')  # 11
        # self.unsample1 = nn.Upsample(scale_factor=2.0, mode='nearest')  # 15
        #
        # self.concat0 = Concat()  # 12
        # self.concat1 = Concat()  # 16
        # self.concat2 = Concat()  # 19
        # self.concat3 = Concat()  # 22
        self.detect = Detect(nc=80,
                             anchors=[[3, 4, 4, 8, 7, 6],
                                      [7, 12, 15, 9, 12, 18],
                                      [27, 15, 23, 29, 46, 36]],
                             ch=(64, 128, 256))

        self.detect1 = Detect(nc=80,
                             anchors=[[3, 4, 4, 8, 7, 6],
                                      ],
                             ch=(64, 128, 256))

    def forward(self, x):
        # backbone x[b, c, w, h]
        p1 = self.conv0(x)            # [b, 16, w/2, h/2]
        p2 = self.conv1(p1)           # [b, 32, w/4, h/4]
        x  = self.c0(p2)              # [b, 32, w/4, h/4]
        p3 = self.conv2(x)            # [b, 64, w/8, h/8]
        x  = self.c1(p3)              # [b, 64, w/8, h/8]
        p4 = self.conv3(x)            # [b, 128, w/16, h/16]
        x  = self.c2(p4)              # [b, 128, w/16, h/16]
        p5 = self.conv4(x)            # [b, 256, w/32, h/32]
        x  = self.c3(p5)              # [b, 256, w/32, h/32]
        x  = self.sppf0(x)            # [b, 256, w/32, h/32]

        # head
        p10 = self.conv5(x)           # [b, 128, w/32, h/32]
        # print(p10.shape)
        x   = self.unsample0(p10)     # [b, 128, w/16, h/16]
        # print(x.shape)
        x   = self.concat0((x, p4))   # [b, 256, w/16, h/16]
        x   = self.c4(x)              # [b, 128, w/16, h/16]
        p14 = self.conv6(x)           # [b, 64, w/16, h/16]
        x   = self.unsample1(p14)     # [b, 64, w/8, h/8]
        x   = self.concat1((x, p3))   # [b, 128, w/8, h/8]
        p17 = self.c5(x)              # [b, 64, w/8, h/8]

        # x   = self.conv7(p17)         # [b, 64, w/16, h/16]
        # x   = self.concat2((x, p14))  # [b, 128, w/16, h/16]
        # p20 = self.c6(x)              # [b, 128, w/16, h/16]
        # x   = self.conv8(p20)         # [b, 128, w/32, h/32]
        # x   = self.concat3((x, p10))  # [b, 128, w/32, h/32]
        # p23 = self.c7(x)              # [b, 256, w/32, h/32]

        # y = self.detect([p17, p20, p23])
        y = self.detect1([p17])

        return y

def info(model, img_size):

    inputs = torch.randn(1, 1, img_size, img_size)
    flops = profile(model, (inputs,)) / 1E9 * 2
    return flops


if __name__ == '__main__':
    input_tensor = torch.randn(16, 1, 208, 208)
    model = YOLOV5n()

    # print(model)
    y = model(input_tensor)
    t = time.time()
    for _ in range(100):
        y = model(input_tensor)
    print((time.time() - t) * 10/32)  # so here we no need to plus 1000

    print(y[0].shape)