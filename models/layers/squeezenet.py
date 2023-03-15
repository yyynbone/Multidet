import torch
import torch.nn as nn

class Fire(nn.Module):
    def __init__(self, in_channel,squeeze_channel, expand1x1_planes, expand3x3_planes):
        super().__init__()
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channel, squeeze_channel, 1),
            nn.BatchNorm2d(squeeze_channel),
            nn.ReLU(inplace=True)
        )

        self.expand_1x1 = nn.Sequential(
            nn.Conv2d(squeeze_channel, expand1x1_planes, 1),
            nn.BatchNorm2d(expand1x1_planes),
            nn.ReLU(inplace=True)
        )

        self.expand_3x3 = nn.Sequential(
            nn.Conv2d(squeeze_channel, expand3x3_planes, 3, padding=1),
            nn.BatchNorm2d(expand3x3_planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.squeeze(x)
        x = torch.cat([
            self.expand_1x1(x),
            self.expand_3x3(x)
        ], 1)
        return x

class FireM(nn.Module):
    def __init__(self, in_channel, squeeze_channel, expand1x1_planes, expand3x3_planes, max_pool=True):
        super().__init__()
        # if max_pool:
        #     self.m = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))
        # else:
        #     self.m = nn.Sequential()
        # # torch 1.10 has no attribute of append
        # self.m.append(Fire(in_channel, squeeze_channel, expand1x1_planes, expand3x3_planes))
        # self.m.append(Fire(expand1x1_planes+expand3x3_planes, squeeze_channel, expand1x1_planes, expand3x3_planes))
        if max_pool:
            self.m = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                                   Fire(in_channel, squeeze_channel, expand1x1_planes, expand3x3_planes),
                                   Fire(expand1x1_planes+expand3x3_planes, squeeze_channel, expand1x1_planes, expand3x3_planes))
        else:
            self.m = nn.Sequential(Fire(in_channel, squeeze_channel, expand1x1_planes, expand3x3_planes),
                                   Fire(expand1x1_planes+expand3x3_planes, squeeze_channel, expand1x1_planes, expand3x3_planes))

    def forward(self,x):
        return self.m(x)


