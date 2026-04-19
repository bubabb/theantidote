"""Table 3: Number-of-copies ablation for Antidote.

Sweeps k in {2, 4, 8, 16}. Each k requires re-running the attack and
re-extracting features since the dataset changes.
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

    rows = []
    summary = {}
    for k in cfg["sweeps"]["copies"]:
        print(f"\n--- k={k} copies ---")
        attack_feat = TruthSerumAttack(
            bundle.trainset_features,
            n_targets=cfg["attack"]["n_targets"],
            n_copies=int(k),
            seed=cfg["seed"],
        )
        poisoned_feat = attack_feat.create_poisoned_dataset()
        gt = poisoned_feat.get_poison_mask()

        defense = AntidoteDefense(
            similarity_threshold=cfg["defense"]["similarity_threshold"],
            device=device,
        )
        features, _ = defense.extract_features(poisoned_feat)
        labels_arr = np.asarray(
            [poisoned_feat[i][1] for i in range(len(poisoned_feat))]
        )
        pred = defense.detect(labels_arr, features)
        metrics = evaluate_detection(pred, gt)
        print(f"  {metrics}")
        rows.append({"copies": int(k), **metrics})
        summary[str(k)] = metrics

    results_dir = Path(cfg["output"]["results_dir"])
    save_csv_rows(rows, results_dir / "table3_copies_ablation.csv")
    save_json(summary, results_dir / "table3_copies_ablation.json")


if __name__ == "__main__":
    main()
