"""Table 1: Detection performance (Antidote vs Spectral Signatures).

Runs the main attack (250 targets x 8 copies), extracts pretrained features,
runs Antidote detection at threshold 0.99, trains a poisoned model for the
Spectral Signatures baseline, and writes results to results/table1_detection.*
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from antidote import (
    AntidoteDefense,
    SpectralSignaturesDefense,
    TruthSerumAttack,
    create_resnet18_cifar,
    evaluate_detection,
    train_model,
)
from antidote.data import cifar10_datasets
from antidote.metrics import seed_everything
from antidote.models import get_device

from _common import ensure_dirs, load_config, save_csv_rows, save_json


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    seed_everything(cfg["seed"])
    device = get_device()
    print(f"device: {device}")

    bundle = cifar10_datasets(root=cfg["data"]["root"], download=cfg["data"]["download"])

    # Attack on the training view (for model training) and feature view (for detection)
    attack_train = TruthSerumAttack(
        bundle.trainset,
        n_targets=cfg["attack"]["n_targets"],
        n_copies=cfg["attack"]["n_copies"],
        seed=cfg["seed"],
    )
    attack_feat = TruthSerumAttack(
        bundle.trainset_features,
        n_targets=cfg["attack"]["n_targets"],
        n_copies=cfg["attack"]["n_copies"],
        seed=cfg["seed"],
    )
    poisoned_train = attack_train.create_poisoned_dataset()
    poisoned_feat = attack_feat.create_poisoned_dataset()
    gt = poisoned_train.get_poison_mask()
    print(f"total={len(poisoned_train)}  poisons={int(gt.sum())}")

    # Antidote detection
    defense = AntidoteDefense(
        similarity_threshold=cfg["defense"]["similarity_threshold"],
        cluster_size_threshold=cfg["defense"]["cluster_size_threshold"],
        feature_batch_size=cfg["defense"]["feature_batch_size"],
        similarity_batch_size=cfg["defense"]["similarity_batch_size"],
        device=device,
    )
    features, _ = defense.extract_features(poisoned_feat)
    labels_arr = np.asarray(
        [poisoned_train[i][1] for i in range(len(poisoned_train))]
    )
    antidote_pred = defense.detect(labels_arr, features)
    antidote_metrics = evaluate_detection(antidote_pred, gt)
    print(f"Antidote:  {antidote_metrics}")

    # Spectral Signatures baseline: train poisoned model first
    print("\ntraining poisoned model for Spectral baseline...")
    poisoned_loader = torch.utils.data.DataLoader(
        poisoned_train,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["eval"]["num_workers"],
    )
    poisoned_model = train_model(
        create_resnet18_cifar(),
        poisoned_loader,
        epochs=cfg["training"]["epochs"],
        lr=cfg["training"]["lr"],
        momentum=cfg["training"]["momentum"],
        weight_decay=cfg["training"]["weight_decay"],
        device=device,
    )
    spectral = SpectralSignaturesDefense(percentile=95.0)
    spectral_pred = spectral.detect(
        poisoned_model,
        poisoned_train,
        device=device,
        batch_size=cfg["eval"]["test_batch_size"],
        num_workers=cfg["eval"]["num_workers"],
    )
    spectral_metrics = evaluate_detection(spectral_pred, gt)
    print(f"Spectral:  {spectral_metrics}")

    # Persist
    results_dir = Path(cfg["output"]["results_dir"])
    rows = [
        {"method": "Antidote", **antidote_metrics},
        {"method": "SpectralSignatures", **spectral_metrics},
    ]
    save_csv_rows(rows, results_dir / "table1_detection.csv")
    save_json(
        {"antidote": antidote_metrics, "spectral_signatures": spectral_metrics},
        results_dir / "table1_detection.json",
    )


if __name__ == "__main__":
    main()
