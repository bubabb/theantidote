"""Membership inference scoring.

Truth Serum inverts standard MIA: targets have HIGH loss (not low) because the
model learned conflicting labels from the poison copies. We therefore score
samples by raw loss, not negative loss.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


@torch.no_grad()
def get_losses(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device | str | None = None,
) -> np.ndarray:
    """Return per-sample cross-entropy losses for every batch in loader."""
    device = torch.device(
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = model.to(device).eval()
    criterion = nn.CrossEntropyLoss(reduction="none")
    losses = []
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        losses.extend(criterion(model(inputs), targets).cpu().numpy())
    return np.asarray(losses)


def compute_mi_truth_serum(
    model: nn.Module,
    target_loader: torch.utils.data.DataLoader,
    nonmember_loader: torch.utils.data.DataLoader,
    device: torch.device | str | None = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Truth Serum MI: HIGH loss on targets signals membership.

    Returns (auc, target_losses, nonmember_losses).
    """
    target_losses = get_losses(model, target_loader, device=device)
    nonmember_losses = get_losses(model, nonmember_loader, device=device)
    scores = np.concatenate([target_losses, nonmember_losses])
    labels = np.concatenate(
        [np.ones(len(target_losses)), np.zeros(len(nonmember_losses))]
    )
    auc = float(roc_auc_score(labels, scores))
    return auc, target_losses, nonmember_losses


def compute_mi_standard(
    model: nn.Module,
    member_loader: torch.utils.data.DataLoader,
    nonmember_loader: torch.utils.data.DataLoader,
    device: torch.device | str | None = None,
) -> float:
    """Standard MI (Shokri et al.): LOW loss on members signals membership."""
    member_losses = get_losses(model, member_loader, device=device)
    nonmember_losses = get_losses(model, nonmember_loader, device=device)
    scores = np.concatenate([-member_losses, -nonmember_losses])
    labels = np.concatenate(
        [np.ones(len(member_losses)), np.zeros(len(nonmember_losses))]
    )
    return float(roc_auc_score(labels, scores))
