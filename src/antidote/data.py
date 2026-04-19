"""CIFAR-10 loading and the poisoned-dataset wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
import torchvision
import torchvision.transforms as T

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def train_transform() -> T.Compose:
    return T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def test_transform() -> T.Compose:
    return T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def feature_transform() -> T.Compose:
    """Transform for the ImageNet-pretrained feature extractor.

    Resizes to 224 and applies ImageNet normalization so the pretrained
    ResNet-18 receives inputs in the distribution it was trained on.
    """
    return T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


@dataclass
class CIFAR10Bundle:
    """Three CIFAR-10 views used across the pipeline."""

    trainset: torchvision.datasets.CIFAR10
    testset: torchvision.datasets.CIFAR10
    trainset_features: torchvision.datasets.CIFAR10


def cifar10_datasets(root: str | Path = "./data", download: bool = True) -> CIFAR10Bundle:
    """Load the three CIFAR-10 views the pipeline needs.

    - `trainset`: augmented 32x32 view used for training the classifier
    - `testset`: clean 32x32 view used for evaluation
    - `trainset_features`: 224x224 ImageNet-normalized view used for detection
    """
    root = str(root)
    return CIFAR10Bundle(
        trainset=torchvision.datasets.CIFAR10(
            root=root, train=True, download=download, transform=train_transform()
        ),
        testset=torchvision.datasets.CIFAR10(
            root=root, train=False, download=download, transform=test_transform()
        ),
        trainset_features=torchvision.datasets.CIFAR10(
            root=root, train=True, download=download, transform=feature_transform()
        ),
    )


class PoisonedDataset(torch.utils.data.Dataset):
    """Training dataset with clean + poison samples and a ground-truth mask."""

    def __init__(
        self,
        images: Sequence[torch.Tensor],
        labels: Sequence[int],
        is_poison: Sequence[bool],
        original_indices: Sequence[int],
    ) -> None:
        self.images: List[torch.Tensor] = list(images)
        self.labels: List[int] = [int(y) for y in labels]
        self.is_poison: List[bool] = [bool(p) for p in is_poison]
        self.original_indices: List[int] = [int(i) for i in original_indices]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        return self.images[idx], self.labels[idx]

    def get_poison_mask(self) -> np.ndarray:
        return np.asarray(self.is_poison, dtype=bool)

    def filter(self, keep_mask: np.ndarray) -> "PoisonedDataset":
        """Return a new PoisonedDataset containing only samples where keep_mask is True."""
        idx = np.where(keep_mask)[0]
        return PoisonedDataset(
            images=[self.images[i] for i in idx],
            labels=[self.labels[i] for i in idx],
            is_poison=[self.is_poison[i] for i in idx],
            original_indices=[self.original_indices[i] for i in idx],
        )
