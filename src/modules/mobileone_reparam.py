import torch
import torch.nn as nn


class MobileOneReparam:
    @staticmethod
    def fuse_conv_bn(conv, bn):
        w = conv.weight
        mean = bn.running_mean
        var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps

        std = torch.sqrt(var + eps)
        scale = gamma / std

        fused_w = w * scale.reshape(-1, 1, 1, 1)
        fused_b = beta - mean * scale

        return fused_w, fused_b

    @staticmethod
    def fuse_block(block):
        fused_weight = None
        fused_bias = None

        w, b = MobileOneReparam.fuse_conv_bn(block.conv.conv, block.conv.bn)
        fused_weight, fused_bias = w, b

        for br in block.branches:
            w, b = MobileOneReparam.fuse_conv_bn(br.conv, br.bn)
            fused_weight += w
            fused_bias += b

        if block.identity is not None:
            id_conv = block.identity.fuse_to_conv(kernel_size=3)
            fused_weight += id_conv.weight
            fused_bias += id_conv.bias

        return fused_weight, fused_bias
