"""Antidote: near-duplicate detection using pretrained ImageNet features.

The attack's unavoidable signature is *near-duplicate samples with inconsistent
labels*. We detect them using features from a ResNet-18 pretrained on ImageNet
— crucially, a model that has never seen the potentially poisoned training set,
which breaks the circular dependency of using a corrupted model to find
corruption.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm


class AntidoteDefense:
    """Detect Truth Serum poisons via near-duplicate + label-mismatch signals.

    Two flags combine into the suspect set:
      1. Label mismatch: sample has a near-duplicate with a different label
      2. Cluster size: sample has 3+ near-duplicates (poison copy cluster)
    """

    def __init__(
        self,
        similarity_threshold: float = 0.99,
        cluster_size_threshold: int = 3,
        device: torch.device | str | None = None,
        feature_batch_size: int = 256,
        similarity_batch_size: int = 2000,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.cluster_size_threshold = cluster_size_threshold
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.feature_batch_size = feature_batch_size
        self.similarity_batch_size = similarity_batch_size
        self.feature_extractor = self._build_feature_extractor()

    def _build_feature_extractor(self) -> nn.Module:
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Identity()
        model.eval()
        return model.to(self.device)

    def extract_features(
        self, dataset: torch.utils.data.Dataset, num_workers: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract 512-dim pretrained features for every sample in the dataset."""
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.feature_batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        features, labels = [], []
        with torch.no_grad():
            for imgs, lbls in tqdm(loader, desc="Extracting features"):
                feats = self.feature_extractor(imgs.to(self.device)).cpu().numpy()
                features.append(feats)
                labels.append(lbls.numpy())
        return np.vstack(features), np.concatenate(labels)

    def detect(self, labels: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Flag suspected-poison indices.

        Returns a boolean array of length n, True at flagged indices.
        """
        n = len(features)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features_norm = features / (norms + 1e-8)
        suspected = np.zeros(n, dtype=bool)

        for i in tqdm(
            range(0, n, self.similarity_batch_size),
            desc=f"Detecting (tau={self.similarity_threshold})",
        ):
            end_i = min(i + self.similarity_batch_size, n)
            sims = features_norm[i:end_i] @ features_norm.T
            for j, row_idx in enumerate(range(i, end_i)):
                row = sims[j].copy()
                row[row_idx] = 0.0  # exclude self
                dup_idx = np.where(row > self.similarity_threshold)[0]
                if dup_idx.size == 0:
                    continue
                if np.any(labels[dup_idx] != labels[row_idx]):
                    suspected[row_idx] = True
                if dup_idx.size >= self.cluster_size_threshold:
                    suspected[row_idx] = True

        return suspected
