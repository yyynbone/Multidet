import torch
import torch.nn as nn
from models.layers.common_layer import Conv

class PoolConv(nn.Module):
    def __init__(self, in_channel, out_channel, stride=2):
        super().__init__()

        self.p = nn.MaxPool2d(kernel_size=stride, stride=stride, ceil_mode=True)
        self.m = Conv(in_channel, out_channel, 3, 1)

    def forward(self,x):
        x = self.p(x)
        x = self.m(x)
        return x

class Mulcat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, xs):
        return torch.cat(xs, self.d)
        # x_c = xs[0]
        # for x in xs[1:]:
        #     bs, c, h, w = x.shape
        #     _, u_c, u_h, u_w = x_c.shape
        #     f_bais = h * w - u_h * u_w
        #     if f_bais > 0:
        #         pad = torch.zeros((bs, u_c, f_bais), dtype=x_c.dtype, device=x_c.device)
        #         new = torch.concat([x_c.reshape(bs, u_c, -1), pad], dim=-1)
        #         x_c = new.reshape(bs, u_c, h, w)
        #     elif f_bais < 0:
        #         new = x_c.reshape(bs, u_c, -1)[:, :, :h * w]
        #         x_c = new.reshape(bs, u_c, h, w)
        #     x_c = torch.cat([x_c, x], self.d)
        # return x_c


class MultiDecoder(nn.Module):
    def __init__(self, in_channel, out_channel,  up_sample, up_mode='bilinear'):
        super().__init__()
        self.layers = []
        pool_conv = len(in_channel)-up_sample-1
        for i in range(up_sample):
            self.layers.append( nn.Sequential(nn.Upsample(scale_factor=2**(up_sample-i), mode=up_mode),
                                  Conv(in_channel[i], out_channel, k=3, s=1)) )
        self.layers.append(Conv(in_channel[up_sample], out_channel, k=3, s=1))
        for i in range(pool_conv):
            self.layers.append(PoolConv(in_channel[i+up_sample+1], out_channel, stride=2**(i+1)))
        self.m = nn.Sequential(*self.layers)
        self.cat = Mulcat()

    def forward(self, xs):
        x_d = []
        for i, x in enumerate(xs):
            x_d.append(self.m[i](x))
        x_d = self.cat(x_d)
        return x_d




