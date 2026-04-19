"""Table 4: End-to-end defense effectiveness.

Trains three models — clean, poisoned, defended (Antidote-filtered) — and
measures membership inference AUC on the 250 targets using the corrected
Truth Serum MI signal (high target loss = likely target).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torchvision

from antidote import (
    AntidoteDefense,
    TruthSerumAttack,
    compute_mi_truth_serum,
    create_resnet18_cifar,
    evaluate_accuracy,
    train_model,
)
from antidote.data import cifar10_datasets, test_transform
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

    # Antidote detection -> defended dataset
    defense = AntidoteDefense(
        similarity_threshold=cfg["defense"]["similarity_threshold"],
        device=device,
    )
    features, _ = defense.extract_features(poisoned_feat)
    labels_arr = np.asarray(
        [poisoned_train[i][1] for i in range(len(poisoned_train))]
    )
    suspected = defense.detect(labels_arr, features)
    defended_train = poisoned_train.filter(~suspected)
    print(
        f"removed {int(suspected.sum())} samples; "
        f"{len(defended_train)} remain for defended training"
    )

    # Loaders
    testloader = torch.utils.data.DataLoader(
        bundle.testset,
        batch_size=cfg["eval"]["test_batch_size"],
        shuffle=False,
        num_workers=cfg["eval"]["num_workers"],
    )
    clean_loader = torch.utils.data.DataLoader(
        bundle.trainset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["eval"]["num_workers"],
    )
    poisoned_loader = torch.utils.data.DataLoader(
        poisoned_train,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["eval"]["num_workers"],
    )
    defended_loader = torch.utils.data.DataLoader(
        defended_train,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["eval"]["num_workers"],
    )

    # Train three models
    def _train_fresh(loader):
        seed_everything(cfg["seed"])  # identical init for a fair comparison
        return train_model(
            create_resnet18_cifar(),
            loader,
            epochs=cfg["training"]["epochs"],
            lr=cfg["training"]["lr"],
            momentum=cfg["training"]["momentum"],
            weight_decay=cfg["training"]["weight_decay"],
            device=device,
        )

    print("\n[1/3] training clean model...")
    clean_model = _train_fresh(clean_loader)
    clean_acc = evaluate_accuracy(clean_model, testloader, device=device)

    print("\n[2/3] training poisoned model...")
    poisoned_model = _train_fresh(poisoned_loader)
    poisoned_acc = evaluate_accuracy(poisoned_model, testloader, device=device)

    print("\n[3/3] training defended model...")
    defended_model = _train_fresh(defended_loader)
    defended_acc = evaluate_accuracy(defended_model, testloader, device=device)

    # MI evaluation: query with target images + TRUE labels, using the
    # non-augmented CIFAR transform so measurements are deterministic.
    trainset_clean = torchvision.datasets.CIFAR10(
        root=cfg["data"]["root"],
        train=True,
        download=False,
        transform=test_transform(),
    )
    target_loader = attack_train.target_loader(
        trainset_clean, batch_size=cfg["eval"]["test_batch_size"]
    )

    clean_auc, clean_t, clean_nm = compute_mi_truth_serum(
        clean_model, target_loader, testloader, device=device
    )
    poisoned_auc, poisoned_t, poisoned_nm = compute_mi_truth_serum(
        poisoned_model, target_loader, testloader, device=device
    )
    defended_auc, defended_t, defended_nm = compute_mi_truth_serum(
        defended_model, target_loader, testloader, device=device
    )

    amplification = poisoned_auc - clean_auc
    reduction = poisoned_auc - defended_auc
    effectiveness = (reduction / amplification * 100) if amplification > 0 else 0.0

    rows = [
        {
            "model": "Clean",
            "test_acc": clean_acc,
            "mi_auc": clean_auc,
            "target_loss_mean": float(clean_t.mean()),
            "nonmember_loss_mean": float(clean_nm.mean()),
        },
        {
            "model": "Poisoned",
            "test_acc": poisoned_acc,
            "mi_auc": poisoned_auc,
            "target_loss_mean": float(poisoned_t.mean()),
            "nonmember_loss_mean": float(poisoned_nm.mean()),
        },
        {
            "model": "Defended",
            "test_acc": defended_acc,
            "mi_auc": defended_auc,
            "target_loss_mean": float(defended_t.mean()),
            "nonmember_loss_mean": float(defended_nm.mean()),
        },
    ]
    summary = {
        "results": rows,
        "attack_amplification_auc": amplification,
        "defense_reduction_auc": reduction,
        "effectiveness_pct": effectiveness,
    }

    print(f"\n{'model':<10}  acc     mi_auc  target_loss")
    for r in rows:
        print(
            f"{r['model']:<10}  {r['test_acc']*100:5.1f}%  "
            f"{r['mi_auc']:.3f}   {r['target_loss_mean']:.3f}"
        )
    print(f"\nattack amplification: +{amplification:.3f}")
    print(f"defense reduction:    -{reduction:.3f}")
    print(f"effectiveness:        {effectiveness:.1f}%")

    results_dir = Path(cfg["output"]["results_dir"])
    save_csv_rows(rows, results_dir / "table4_end_to_end.csv")
    save_json(summary, results_dir / "table4_end_to_end.json")

    # Save raw loss arrays for the hero figure
    np.savez(
        results_dir / "table4_losses.npz",
        clean_targets=clean_t,
        clean_nonmembers=clean_nm,
        poisoned_targets=poisoned_t,
        poisoned_nonmembers=poisoned_nm,
        defended_targets=defended_t,
        defended_nonmembers=defended_nm,
    )
    print(f"  wrote {results_dir / 'table4_losses.npz'}")


if __name__ == "__main__":
    main()
