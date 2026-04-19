import torch
import torch.nn as nn


class IdentityBN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        return self.bn(x)

    def fuse_to_conv(self, kernel_size):
        conv = nn.Conv2d(
            self.bn.num_features,
            self.bn.num_features,
            kernel_size,
            padding=kernel_size // 2,
            groups=self.bn.num_features,
            bias=True
        )

        w = torch.zeros_like(conv.weight)
        center = kernel_size // 2

        for i in range(w.size(0)):
            w[i, 0, center, center] = 1.0

        mean = self.bn.running_mean
        var = self.bn.running_var
        gamma = self.bn.weight
        beta = self.bn.bias
        eps = self.bn.eps

        std = torch.sqrt(var + eps)
        scale = gamma / std

        conv.weight.data = w * scale.reshape(-1, 1, 1, 1)
        conv.bias.data = beta - mean * scale

        return conv
