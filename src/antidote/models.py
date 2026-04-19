"""Model architectures used throughout Antidote."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18


def create_resnet18_cifar(num_classes: int = 10) -> nn.Module:
    """ResNet-18 adapted for 32x32 inputs.

    Replaces the stock 7x7 stride-2 conv with a 3x3 stride-1 conv and removes
    the initial max-pool to preserve spatial resolution on CIFAR-sized images.
    """
    model = resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def get_device(prefer: str | None = None) -> torch.device:
    """Return a torch.device: GPU if available, else CPU.

    Pass prefer="cpu" to force CPU.
    """
    if prefer == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
