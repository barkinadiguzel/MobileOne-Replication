import torch
import torch.nn as nn
from blocks.depthwise_conv import DepthwiseConvBN
from blocks.pointwise_conv import PointwiseConvBN
from blocks.identity_bn import IdentityBN


class MobileOneBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=4):
        super().__init__()

        self.k = k

        self.dw = DepthwiseConvBN(in_ch)
        self.pw = PointwiseConvBN(in_ch, out_ch)

        self.branches = nn.ModuleList([
            DepthwiseConvBN(in_ch) for _ in range(k)
        ])

        self.identity = IdentityBN(in_ch)

        self.act = nn.ReLU()

    def forward(self, x):
        out = self.dw(x)

        for b in self.branches:
            out = out + b(x)

        out = out + self.identity(x)

        out = self.pw(out)

        return self.act(out)

    def reparameterize(self):
        # placeholder
        return self
