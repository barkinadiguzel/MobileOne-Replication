import torch
import torch.nn as nn
from blocks.conv_bn import ConvBN
from blocks.identity_bn import IdentityBN


class MobileOneBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, k=4, stride=1):
        super().__init__()

        padding = kernel_size // 2
        self.k = k

        self.conv = ConvBN(in_ch, out_ch, kernel_size, stride=stride, padding=padding)

        self.branches = nn.ModuleList([
            ConvBN(in_ch, out_ch, kernel_size, stride=stride, padding=padding)
            for _ in range(k)
        ])

        self.identity = IdentityBN(in_ch) if in_ch == out_ch else None

        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)

        for b in self.branches:
            out = out + b(x)

        if self.identity is not None:
            out = out + self.identity(x)

        return self.act(out)
