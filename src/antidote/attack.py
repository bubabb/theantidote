"""Truth Serum data-poisoning attack (Tramer et al., CCS 2022).

The attacker selects n_targets samples, copies each one n_copies times with a
wrong label, and inserts all copies into the training set. After training, the
target samples exhibit elevated loss due to the label conflict — which makes
them trivially identifiable via membership inference.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from antidote.data import PoisonedDataset


class TruthSerumAttack:
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        n_targets: int = 250,
        n_copies: int = 8,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.n_targets = n_targets
        self.n_copies = n_copies
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.target_indices: np.ndarray = rng.choice(
            len(dataset), size=n_targets, replace=False
        )

    def create_poisoned_dataset(self) -> PoisonedDataset:
        images, labels, is_poison, original_indices = [], [], [], []

        for i in range(len(self.dataset)):
            img, label = self.dataset[i]
            images.append(img)
            labels.append(label)
            is_poison.append(False)
            original_indices.append(i)

        # Use a derived RNG so the wrong-label draw is reproducible and
        # independent of target selection
        rng = np.random.default_rng(self.seed + 1)
        for target_idx in self.target_indices:
            img, true_label = self.dataset[int(target_idx)]
            wrong_label = (int(true_label) + int(rng.integers(1, 10))) % 10
            for _ in range(self.n_copies):
                images.append(img.clone() if isinstance(img, torch.Tensor) else img)
                labels.append(wrong_label)
                is_poison.append(True)
                original_indices.append(int(target_idx))

        return PoisonedDataset(images, labels, is_poison, original_indices)

    def target_loader(
        self, clean_dataset: torch.utils.data.Dataset, batch_size: int = 256
    ) -> torch.utils.data.DataLoader:
        """Build a DataLoader over the target samples with their TRUE labels.

        Used for membership inference: we want to query the trained model with
        the *original* target image + label, not the poisoned (x, y') version.
        """
        samples = [clean_dataset[int(i)] for i in self.target_indices]
        images = torch.stack([s[0] for s in samples])
        targets = torch.tensor([s[1] for s in samples])
        ds = torch.utils.data.TensorDataset(images, targets)
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
