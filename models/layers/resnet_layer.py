import torch
from .common_layer import Conv, ReluConv
import torch.nn as nn
# -------------------------------------------------------------------------
class BasicBlock(nn.Module):
    """BasicBlock for ResNet.

    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        stride (int): stride of the block. Default: 1
        downsample (nn.Module, optional): downsample operation on identity
            branch. Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 k=3,
                 stride=1,
                 p=1
                 ):
        super(BasicBlock, self).__init__()

        self.conv1 = ReluConv(in_channels, out_channels, k, stride, p)

        self.conv2 = Conv(out_channels, out_channels, k, 1, p, act=False)

        self.relu = nn.ReLU(inplace=True)

        if stride !=1 or in_channels != out_channels:
            self.downsample = Conv(in_channels, out_channels, 1, stride, act=False) # padding=0
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Repeat_BasicBlock(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 n=1,
                 k=3,
                 s=1,
                 p=1
                 ):
        super().__init__()
        self.m = nn.Sequential( *(BasicBlock(in_channels, out_channels, k, s, p) for _ in range(n)) )

    def forward(self, x):
        return self.m(x)


