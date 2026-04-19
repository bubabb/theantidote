"""Table 2: Similarity threshold ablation for Antidote.

Sweeps tau in {0.95, 0.99, 0.999} on the same feature matrix (no retraining).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from antidote import AntidoteDefense, TruthSerumAttack, evaluate_detection
from antidote.data import cifar10_datasets
from antidote.metrics import seed_everything
from antidote.models import get_device

from _common import ensure_dirs, load_config, save_csv_rows, save_json


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    seed_everything(cfg["seed"])
    device = get_device()

    bundle = cifar10_datasets(root=cfg["data"]["root"], download=cfg["data"]["download"])
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

    # Extract features once, sweep thresholds
    defense = AntidoteDefense(
        similarity_threshold=cfg["defense"]["similarity_threshold"],
        device=device,
    )
    features, _ = defense.extract_features(poisoned_feat)
    labels_arr = np.asarray(
        [poisoned_train[i][1] for i in range(len(poisoned_train))]
    )

    rows = []
    summary = {}
    for tau in cfg["sweeps"]["thresholds"]:
        defense.similarity_threshold = float(tau)
        pred = defense.detect(labels_arr, features)
        metrics = evaluate_detection(pred, gt)
        print(f"tau={tau}  {metrics}")
        rows.append({"threshold": tau, **metrics})
        summary[str(tau)] = metrics

    results_dir = Path(cfg["output"]["results_dir"])
    save_csv_rows(rows, results_dir / "table2_threshold_ablation.csv")
    save_json(summary, results_dir / "table2_threshold_ablation.json")


if __name__ == "__main__":
    main()
