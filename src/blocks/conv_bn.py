import torch
import torch.nn as nn


class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, groups=1):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False
        )

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))

    def fuse(self):

        fused_conv = nn.Conv2d(
            self.conv.in_channels,
            self.conv.out_channels,
            self.conv.kernel_size,
            stride=self.conv.stride,
            padding=self.conv.padding,
            groups=self.conv.groups,
            bias=True
        )

        w = self.conv.weight
        mean = self.bn.running_mean
        var = self.bn.running_var
        gamma = self.bn.weight
        beta = self.bn.bias
        eps = self.bn.eps

        std = torch.sqrt(var + eps)
        scale = gamma / std

        fused_conv.weight.data = w * scale.reshape(-1, 1, 1, 1)

        if self.conv.bias is None:
            bias = torch.zeros(w.size(0))
        else:
            bias = self.conv.bias

        fused_conv.bias.data = beta + (bias - mean) * scale

        return fused_conv
