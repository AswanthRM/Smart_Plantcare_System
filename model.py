# model.py

import torch
import torch.nn as nn
import torchvision.models as models


class SiameseNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Load MobileNetV2 backbone
        base = models.mobilenet_v2(weights="DEFAULT")

        # Feature extractor
        self.features = base.features

        # Global Average Pooling
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Embedding Head
        self.fc = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128)
        )

    def forward_once(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        return out1, out2