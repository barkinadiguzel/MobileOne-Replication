import torch.nn as nn
from modules.stage_builder import StageBuilder
from modules.mobileone_reparam import MobileOneReparam


class MobileOne(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.stage1 = StageBuilder(64, 64, 2, k=4)
        self.stage2 = StageBuilder(64, 128, 2, k=4)
        self.stage3 = StageBuilder(128, 256, 6, k=4)
        self.stage4 = StageBuilder(256, 512, 2, k=4)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.pool(x).flatten(1)
        return self.fc(x)

    def reparameterize_model(self):
        for m in self.modules():
            if hasattr(m, "conv") and hasattr(m, "branches"):

                w, b = MobileOneReparam.fuse_block(m)

                fused_conv = nn.Conv2d(
                    w.shape[1],
                    w.shape[0],
                    3,
                    padding=1,
                    bias=True
                )

                fused_conv.weight.data = w
                fused_conv.bias.data = b

                m.conv = fused_conv
                m.branches = nn.ModuleList()
                m.identity = None

        return self
