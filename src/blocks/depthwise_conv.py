import torch.nn as nn


class DepthwiseConvBN(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=channels,
            bias=False
        )
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        return self.bn(self.conv(x))

    def fuse(self):
        return self
