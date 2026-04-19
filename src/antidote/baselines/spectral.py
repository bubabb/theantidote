"""Spectral Signatures backdoor defense (Tran et al., NeurIPS 2018).

Included as a baseline. Spectral Signatures is designed for *trigger-based*
backdoors where poisoned samples are feature-space outliers. It is not
designed for label-flipping attacks like Truth Serum, where poisons and
legitimate samples are visually identical — only labels differ. The empirical
result in the paper (F1=0.010) reflects this mismatch.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class SpectralSignaturesDefense:
    """Per-class top-eigenvector outlier scoring on learned representations."""

    def __init__(self, percentile: float = 95.0, num_classes: int = 10) -> None:
        self.percentile = percentile
        self.num_classes = num_classes

    def detect(
        self,
        model: nn.Module,
        dataset: torch.utils.data.Dataset,
        device: torch.device | str | None = None,
        batch_size: int = 256,
        num_workers: int = 2,
    ) -> np.ndarray:
        device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        model = model.to(device).eval()

        activations = {}

        def hook_fn(module, inputs, output):
            activations["rep"] = output.detach()

        hook = model.avgpool.register_forward_hook(hook_fn)

        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        reps, labels_list = [], []
        with torch.no_grad():
            for inputs, targets in tqdm(loader, desc="Spectral extraction"):
                _ = model(inputs.to(device))
                reps.append(activations["rep"].squeeze().cpu().numpy())
                labels_list.append(targets.numpy())
        hook.remove()

        reps = np.vstack(reps)
        labels = np.concatenate(labels_list)
        scores = np.zeros(len(reps))

        for c in range(self.num_classes):
            mask = labels == c
            if mask.sum() < 10:
                continue
            centered = reps[mask] - reps[mask].mean(axis=0)
            try:
                _, _, vt = np.linalg.svd(centered, full_matrices=False)
                scores[mask] = (centered @ vt[0]) ** 2
            except np.linalg.LinAlgError:
                continue

        return scores > np.percentile(scores, self.percentile)
