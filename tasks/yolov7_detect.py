import torch
import torch.nn as nn
import numpy as np
import json

def show_model_param(model,file):
    data={}
    for i,(k, v) in enumerate(model.named_parameters()):
        # print(f"in layer {k}:  param value is:\n{v}")
        data[k] = v.cpu().detach().numpy().tolist()
        print(f'in {k}, {v.shape}')
    with open(file, 'w') as f:
        # sys.stdout = f
        # sys.stderr = f
        json.dump(data, f, indent=4)

def show_state_dict(state_dict,file):
    o_data={}
    for i, (k,v) in enumerate(state_dict.items()):
        if i>400 and i<410:
            o_data[k] = v.cpu().detach().numpy().tolist()
            print(f'in {k}, {v.shape}')
    with open(file, 'w') as f:
        # sys.stdout = f
        # sys.stderr = f
        json.dump(o_data, f, indent=4)

def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    new_csd = {}
    value = list(db.values())
    keys = list(db.keys())
    for i, (k, v) in enumerate(da.items()):
        if not any(x in k for x in exclude) and v.shape == value[i].shape:
            new_csd[keys[i]] = v
        else:
            print(f'{k} shape {v.shape} not match {keys[i]} shape {value[i].shape}')
    return new_csd

##### basic ####

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class RepConv(nn.Module):
    # Represented convolution
    # https://arxiv.org/abs/2101.03697

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True, deploy=False):
        super(RepConv, self).__init__()

        self.deploy = deploy
        self.groups = g
        self.in_channels = c1
        self.out_channels = c2

        assert k == 3
        assert autopad(k, p) == 1

        padding_11 = autopad(k, p) - k // 2

        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

        if deploy:
            self.rbr_reparam = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=True)

        else:
            self.rbr_identity = (nn.BatchNorm2d(num_features=c1) if c2 == c1 and s == 1 else None)

            self.rbr_dense = nn.Sequential(
                nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2),
            )

            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(c1, c2, 1, s, padding_11, groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2),
            )

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.act(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=np.float32
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return (
            kernel.detach().cpu().numpy(),
            bias.detach().cpu().numpy(),
        )

    def fuse_conv_bn(self, conv, bn):

        std = (bn.running_var + bn.eps).sqrt()
        bias = bn.bias - bn.running_mean * bn.weight / std

        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        weights = conv.weight * t

        bn = nn.Identity()
        conv = nn.Conv2d(in_channels=conv.in_channels,
                         out_channels=conv.out_channels,
                         kernel_size=conv.kernel_size,
                         stride=conv.stride,
                         padding=conv.padding,
                         dilation=conv.dilation,
                         groups=conv.groups,
                         bias=True,
                         padding_mode=conv.padding_mode)

        conv.weight = torch.nn.Parameter(weights)
        conv.bias = torch.nn.Parameter(bias)
        return conv

    def fuse_repvgg_block(self):
        if self.deploy:
            return
        print(f"RepConv.fuse_repvgg_block")

        self.rbr_dense = self.fuse_conv_bn(self.rbr_dense[0], self.rbr_dense[1])

        self.rbr_1x1 = self.fuse_conv_bn(self.rbr_1x1[0], self.rbr_1x1[1])
        rbr_1x1_bias = self.rbr_1x1.bias
        weight_1x1_expanded = torch.nn.functional.pad(self.rbr_1x1.weight, [1, 1, 1, 1])

        # Fuse self.rbr_identity
        if (isinstance(self.rbr_identity, nn.BatchNorm2d) or isinstance(self.rbr_identity,
                                                                        nn.modules.batchnorm.SyncBatchNorm)):
            # print(f"fuse: rbr_identity == BatchNorm2d or SyncBatchNorm")
            identity_conv_1x1 = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=self.groups,
                bias=False)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.to(self.rbr_1x1.weight.data.device)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.squeeze().squeeze()
            # print(f" identity_conv_1x1.weight = {identity_conv_1x1.weight.shape}")
            identity_conv_1x1.weight.data.fill_(0.0)
            identity_conv_1x1.weight.data.fill_diagonal_(1.0)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.unsqueeze(2).unsqueeze(3)
            # print(f" identity_conv_1x1.weight = {identity_conv_1x1.weight.shape}")

            identity_conv_1x1 = self.fuse_conv_bn(identity_conv_1x1, self.rbr_identity)
            bias_identity_expanded = identity_conv_1x1.bias
            weight_identity_expanded = torch.nn.functional.pad(identity_conv_1x1.weight, [1, 1, 1, 1])
        else:
            # print(f"fuse: rbr_identity != BatchNorm2d, rbr_identity = {self.rbr_identity}")
            bias_identity_expanded = torch.nn.Parameter(torch.zeros_like(rbr_1x1_bias))
            weight_identity_expanded = torch.nn.Parameter(torch.zeros_like(weight_1x1_expanded))

            # print(f"self.rbr_1x1.weight = {self.rbr_1x1.weight.shape}, ")
        # print(f"weight_1x1_expanded = {weight_1x1_expanded.shape}, ")
        # print(f"self.rbr_dense.weight = {self.rbr_dense.weight.shape}, ")

        self.rbr_dense.weight = torch.nn.Parameter(
            self.rbr_dense.weight + weight_1x1_expanded + weight_identity_expanded)
        self.rbr_dense.bias = torch.nn.Parameter(self.rbr_dense.bias + rbr_1x1_bias + bias_identity_expanded)

        self.rbr_reparam = self.rbr_dense
        self.deploy = True

        if self.rbr_identity is not None:
            del self.rbr_identity
            self.rbr_identity = None

        if self.rbr_1x1 is not None:
            del self.rbr_1x1
            self.rbr_1x1 = None

        if self.rbr_dense is not None:
            del self.rbr_dense
            self.rbr_dense = None

class SPPCSPC(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))

class ImplicitA(nn.Module):
    def __init__(self, channel, mean=0., std=.02):
        super(ImplicitA, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit + x

class ImplicitM(nn.Module):
    def __init__(self, channel, mean=1., std=.02):
        super(ImplicitM, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.ones(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit * x

class IDetect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export
    end2end = False
    include_nms = False
    concat = False

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(IDetect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

        self.ia = nn.ModuleList(ImplicitA(x) for x in ch)
        self.im = nn.ModuleList(ImplicitM(self.no * self.na) for _ in ch)

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](self.ia[i](x[i]))  # conv
            x[i] = self.im[i](x[i])
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))
        print('now training is ', self.training)
        return x if self.training else (torch.cat(z, 1), x)

    def fuseforward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                if not torch.onnx.is_in_onnx_export():
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:
                    xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                    xy = xy * (2. * self.stride[i]) + (self.stride[i] * (self.grid[i] - 0.5))  # new xy
                    wh = wh ** 2 * (4 * self.anchor_grid[i].data)  # new wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))

        if self.training:
            out = x
        elif self.end2end:
            out = torch.cat(z, 1)
        elif self.include_nms:
            z = self.convert(z)
            out = (z,)
        elif self.concat:
            out = torch.cat(z, 1)
        else:
            out = (torch.cat(z, 1), x)

        return out

    def fuse(self):
        print("IDetect.fuse")
        # fuse ImplicitA and Convolution
        for i in range(len(self.m)):
            c1, c2, _, _ = self.m[i].weight.shape
            c1_, c2_, _, _ = self.ia[i].implicit.shape
            self.m[i].bias += torch.matmul(self.m[i].weight.reshape(c1, c2),
                                           self.ia[i].implicit.reshape(c2_, c1_)).squeeze(1)

        # fuse ImplicitM and Convolution
        for i in range(len(self.m)):
            c1, c2, _, _ = self.im[i].implicit.shape
            self.m[i].bias *= self.im[i].implicit.reshape(c2)
            self.m[i].weight *= self.im[i].implicit.transpose(0, 1)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def convert(self, z):
        z = torch.cat(z, 1)
        box = z[:, :, :4]
        conf = z[:, :, 4:5]
        score = z[:, :, 5:]
        score *= conf
        convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                      dtype=torch.float32,
                                      device=z.device)
        box @= convert_matrix
        return (box, score)

class MP(nn.Module):
    def __init__(self, c_in, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)
        self.conv1_2 =Conv(c1=c_in, c2=int(c_in/2), k=1, s=1)
        self.conv2_1 =Conv(c1=c_in, c2=int(c_in/2), k=1, s=1)
        self.conv2_2 =Conv(c1=int(c_in/2), c2=int(c_in/2), k=3, s=2)

    def forward(self, x):
        x1 = self.m(x)
        x1 = self.conv1_2(x1)
        x2 = self.conv2_1(x)
        x2 = self.conv2_2(x2)

        x = torch.cat((x2, x1), dim=1)
        return x

class MP_H(nn.Module):
    def __init__(self, c_in, k=2):
        super(MP_H, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)
        self.conv1_2 =Conv(c1=c_in, c2=int(c_in), k=1, s=1)
        self.conv2_1 =Conv(c1=c_in, c2=int(c_in), k=1, s=1)
        self.conv2_2 =Conv(c1=int(c_in), c2=int(c_in), k=3, s=2)

    def forward(self, x, y):
        x1 = self.m(x)
        x1 = self.conv1_2(x1)
        x2 = self.conv2_1(x)
        x2 = self.conv2_2(x2)

        x = torch.cat((x2, x1, y), dim=1)
        return x

class ELAN(nn.Module):
    def __init__(self, c_in, c_devide):
        super(ELAN, self).__init__()
        self.conv1_1 = Conv(c1=c_in, c2=int(c_in/c_devide), k=1, s=1)
        self.conv2_1 = Conv(c1=c_in, c2=int(c_in/c_devide), k=1, s=1)
        self.conv2_2 = Conv(c1=int(c_in/c_devide), c2=int(c_in/c_devide), k=3, s=1)
        self.conv2_3 = Conv(c1=int(c_in/c_devide), c2=int(c_in/c_devide), k=3, s=1)
        self.conv2_4 = Conv(c1=int(c_in/c_devide), c2=int(c_in/c_devide), k=3, s=1)
        self.conv2_5 = Conv(c1=int(c_in/c_devide), c2=int(c_in/c_devide), k=3, s=1)

        self.conv3_1 = Conv(c1=int(c_in/c_devide)*4, c2=int(c_in/c_devide)*4, k=1, s=1)

    def forward(self, x):
        x1 = self.conv1_1(x)
        x2 = self.conv2_1(x)
        x3 = self.conv2_2(x2)
        x4 = self.conv2_3(x3)
        x5 = self.conv2_4(x4)
        x6 = self.conv2_5(x5)

        x7 =torch.cat((x6, x4, x2, x1), dim=1)
        x = self.conv3_1(x7)
        return x

class ELAN_H(nn.Module):
    def __init__(self, c_in, c_devide):
        super(ELAN_H, self).__init__()
        self.conv1_1 = Conv(c1=c_in, c2=int(c_in/c_devide), k=1, s=1)
        self.conv2_1 = Conv(c1=c_in, c2=int(c_in/c_devide), k=1, s=1)
        self.conv2_2 = Conv(c1=int(c_in/c_devide), c2=int(c_in/c_devide/2), k=3, s=1)
        self.conv2_3 = Conv(c1=int(c_in/c_devide/2), c2=int(c_in/c_devide/2), k=3, s=1)
        self.conv2_4 = Conv(c1=int(c_in/c_devide/2), c2=int(c_in/c_devide/2), k=3, s=1)
        self.conv2_5 = Conv(c1=int(c_in/c_devide/2), c2=int(c_in/c_devide/2), k=3, s=1)

        self.conv3_1 = Conv(c1=int(c_in/c_devide)*4, c2=int(c_in/c_devide), k=1, s=1)

    def forward(self, x):
        x1 = self.conv1_1(x)
        x2 = self.conv2_1(x)
        x3 = self.conv2_2(x2)
        x4 = self.conv2_3(x3)
        x5 = self.conv2_4(x4)
        x6 = self.conv2_5(x5)

        x =torch.cat((x6, x5, x4, x3, x2, x1), dim=1)
        x = self.conv3_1(x)
        return x


class YOLOV7(nn.Module):
    def __init__(self, width=.5 ):
        super(YOLOV7, self).__init__()

        # backbone
        self.Conv_0 = Conv(c1=3, c2=int(32*width), k=3, s=1)
        self.Conv_1 = Conv(c1=int(32*width), c2=int(64*width), k=3, s=2)
        self.Conv_2 = Conv(c1=int(64*width), c2=int(64*width), k=3, s=1)
        self.Conv_3 = Conv(c1=int(64*width), c2=int(128*width), k=3, s=2)

        self.ELAN_1 = ELAN(c_in=int(128*width), c_devide=2)
        self.MP_1 = MP(c_in=int(256*width))

        self.ELAN_2 = ELAN(c_in=int(256*width), c_devide=2)
        self.MP_2 = MP(c_in=int(512*width))

        self.ELAN_3 = ELAN(c_in=int(512*width), c_devide=2)
        self.MP_3 = MP(c_in=int(1024*width))

        self.ELAN_4 = ELAN(c_in=int(1024*width), c_devide=4)


        # head
        self.SPPCSPC = SPPCSPC(c1=int(1024*width), c2=int(512*width), n=1)

        self.Upsample = nn.modules.upsampling.Upsample(size=None, scale_factor=2, mode='nearest')

        self.Conv_4 = Conv(c1=int(512*width), c2=int(256*width), k=1, s=1)
        self.Conv_5 = Conv(c1=int(1024*width), c2=int(256*width), k=1, s=1)
        self.ELAN_H_1 = ELAN_H(c_in=int(512*width), c_devide=2)

        self.Conv_6 = Conv(c1=int(256*width), c2=int(128*width), k=1, s=1)
        self.Conv_7 = Conv(c1=int(512*width), c2=int(128*width), k=1, s=1)
        self.ELAN_H_2 = ELAN_H(c_in=int(256*width), c_devide=2)

        self.MP_H_1 = MP_H(c_in=int(128*width))
        self.ELAN_H_3 = ELAN_H(c_in=int(512*width), c_devide=2)
        self.MP_H_2 = MP_H(c_in=int(256*width))
        self.ELAN_H_4 = ELAN_H(c_in=int(1024*width), c_devide=2)

        self.RepConv_75 = RepConv(c1=int(128*width), c2=int(256*width), k=3, s=1)
        self.RepConv_88 = RepConv(c1=int(256*width), c2=int(512*width), k=3, s=1)
        self.RepConv_101 = RepConv(c1=int(512*width), c2=int(1024*width), k=3, s=1)

        self.IDetect = IDetect(nc=8,
                               anchors=[[12, 16, 19, 36, 40, 28],
                                        [36, 75, 76, 55, 72, 146],
                                        [142, 110, 192, 243, 459, 401]],
                               ch=[int(256*width), int(512*width), int(1024*width)])
        self.IDetect.training = False
        self.stride = self.IDetect.stride = [8, 16, 32]

    def forward(self, x, **kwargs):

        # backbone
        # pred_j = {'before 0': x.cpu().detach().numpy().tolist()}
        x = self.Conv_0(x)
        # pred_j['after 0'] = x.cpu().detach().numpy().tolist()
        # show_model_param(self.Conv_0, 'new_conv0w.json')
        p1 = self.Conv_1(x)  # 1/2
        # pred_j['after 1'] = x.cpu().detach().numpy().tolist()
        # show_model_param(self.Conv_0, 'new_conv1w.json')
        x = self.Conv_2(p1)
        p2 = self.Conv_3(x)  # 1/4x
        x = self.ELAN_1(p2)
        p3 = self.MP_1(x)  # 1/8
        x_24 = self.ELAN_2(p3)
        x = self.MP_2(x_24)  # 1/16
        x_37 = self.ELAN_3(x)
        x = self.MP_3(x_37)  # 1/32
        x = self.ELAN_4(x)

        # head
        x_51 = self.SPPCSPC(x)

        x = self.Conv_4(x_51)
        x = self.Upsample(x)
        branch_2 = self.Conv_5(x_37)
        x = torch.cat((branch_2, x), dim=1)

        x_63 = self.ELAN_H_1(x)

        x = self.Conv_6(x_63)
        x = self.Upsample(x)
        branch_1 = self.Conv_7(x_24)
        x = torch.cat((branch_1, x), dim=1)

        x_75 = self.ELAN_H_2(x)

        x = self.MP_H_1(x_75, x_63)

        x_88 = self.ELAN_H_3(x)

        x = self.MP_H_2(x_88, x_51)

        x_101 = self.ELAN_H_4(x)

        x_102 = self.RepConv_75(x_75)
        x_103 = self.RepConv_88(x_88)
        x_104 = self.RepConv_101(x_101)

        y = self.IDetect([x_102, x_103, x_104])
        # pred_j = {'before rep': x_75.cpu().detach().numpy().tolist()}
        # pred_j['after rep'] = x_102.cpu().detach().numpy().tolist()
        # show_model_param(self.RepConv_75, 'new_w.json')
        # pred_j={'before detect': [p.cpu().detach().numpy().tolist() for p in [x_102, x_103, x_104]] }
        # pred_j['after_detect'] = [pred_i.cpu().detach().numpy().tolist() for pred_i in y[0]]
        # with open('new_pred.json','w+') as f:
        #     json.dump(pred_j, f, indent=4)
        return [y]

def load_and_save_dict():
    weight = 'checkpoints/best.pt'
    ck = torch.load(weight)
    sd = ck['model'].state_dict()
    with open('origin_w.json', 'w') as f:
        # sys.stdout = f
        # sys.stderr = f
        json.dump(sd, f, indent=4)
    new_pt = {'state_dict':sd, 'names':ck['model'].names}
    torch.save(new_pt, './checkpoints/yolov7.pt')

if __name__ == '__main__':
    model = YOLOV7()
    with open('img-255.json', 'r') as f:
        im = json.load(f)['img']
    im = torch.tensor(im, dtype=torch.float32)
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    # load_and_save_dict()
    weight = 'checkpoints/yolov7.pt'
    ck = torch.load(weight)
    o_data = {}
    # show_state_dict(ck['state_dict'], 'origin_w.json')
    csd = intersect_dicts(ck['state_dict'], model.state_dict())  # intersect
    model.load_state_dict(csd, strict=False)
    # show_state_dict(model.state_dict(), 'new.json')
    print(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weight}')
    with torch.no_grad():
        model.half()
        model.cuda(device = device)
        model.eval()
        pred, head_out = model(im)[0]
    print("####################")
    print(pred.shape)
    print('done')