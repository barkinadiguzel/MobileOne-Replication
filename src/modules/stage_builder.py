import torch.nn as nn
from modules.mobileone_block import MobileOneBlock


class StageBuilder(nn.Module):
    def __init__(self, in_ch, out_ch, num_blocks, k=4):
        super().__init__()

        layers = []
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            layers.append(MobileOneBlock(in_ch, out_ch, k=k, stride=stride))
            in_ch = out_ch

        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        return self.blocks(x)
