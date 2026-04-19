"""Detection metric helpers."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def evaluate_detection(
    predicted: np.ndarray, ground_truth: np.ndarray
) -> Dict[str, float]:
    """Precision, recall, F1 for a binary detection mask."""
    predicted = np.asarray(predicted, dtype=bool)
    ground_truth = np.asarray(ground_truth, dtype=bool)
    p, r, f1, _ = precision_recall_fscore_support(
        ground_truth, predicted, average="binary", zero_division=0
    )
    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "tp": int(np.sum(predicted & ground_truth)),
        "fp": int(np.sum(predicted & ~ground_truth)),
        "fn": int(np.sum(~predicted & ground_truth)),
        "tn": int(np.sum(~predicted & ~ground_truth)),
    }


def seed_everything(seed: int) -> None:
    """Deterministic seeding across numpy + torch."""
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
