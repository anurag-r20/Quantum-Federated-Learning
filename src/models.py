"""
Models for the MRI privacy-attack experiments.
"""
import torch
import torch.nn as nn


class MRINet(nn.Module):
    """
    Small CNN for 4-class brain-tumour classification on 1 x img_size x img_size grayscale MRI.

    ~0.5M parameters at 64x64, small enough that every weight can be CKKS-encrypted each round.
    """
    def __init__(self, num_classes: int = 4, img_size: int = 64) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * (img_size // 4) ** 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class EvaluationNet(nn.Module):
    """
    Independent classifier (different architecture, trained centrally on the private data) used only
    to score model-inversion reconstructions, as in Zhang et al. (CVPR 2020).
    """
    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(), nn.MaxPool2d(2))

        self.features = nn.Sequential(block(1, 32), block(32, 64), block(64, 128), block(128, 128))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))
