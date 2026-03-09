"""Simple CNN for FashionMNIST classification.

Architecture:
    Conv2d(1, C, 3, padding=1) -> ReLU -> MaxPool2d(2)
    Conv2d(C, C*2, 3, padding=1) -> ReLU -> MaxPool2d(2)
    Flatten -> Linear(C*2 * 7 * 7, 128) -> ReLU -> Linear(128, num_classes)

where C = hparams["channels"] (default 32).
Input: (batch, 1, 28, 28) -> Output: (batch, num_classes)
"""

import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, hparams, device="cpu"):
        super(SimpleCNN, self).__init__()
        self.hparams = hparams
        self.device = device

        channels = hparams["channels"]
        num_classes = hparams["num_classes"]

        self.features = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 2 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
