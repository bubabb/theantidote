"""Regenerate all paper figures from saved results/*.csv and *.npz.

Runs after the table scripts. Writes PNGs under results/figures/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _common import ensure_dirs, load_config

PALETTE = {
    "antidote": "#2a9d8f",
    "spectral": "#e9c46a",
    "clean": "#2a9d8f",
    "poisoned": "#e63946",
    "defended": "#f4a261",
    "neutral": "#264653",
    "accent": "#e76f51",
}


def plot_threshold(ax, summary: dict) -> None:
    ts = list(summary.keys())
    x = np.arange(len(ts))
    w = 0.25
    ax.bar(
        x - w,
        [summary[t]["precision"] for t in ts],
        w,
        label="Precision",
        color=PALETTE["antidote"],
    )
    ax.bar(
        x,
        [summary[t]["recall"] for t in ts],
        w,
        label="Recall",
        color=PALETTE["accent"],
    )
    ax.bar(
        x + w,
        [summary[t]["f1"] for t in ts],
        w,
        label="F1",
        color=PALETTE["neutral"],
    )
    ax.set_xticks(x)
    ax.set_xticklabels(ts)
    ax.set_xlabel("Similarity threshold")
    ax.set_ylabel("Score")
    ax.set_title("(a) Threshold ablation")
    ax.set_ylim(0, 1.1)
    ax.legend()


def plot_copies(ax, summary: dict) -> None:
    ks = [int(k) for k in summary.keys()]
    ks.sort()
    ax.plot(
        ks,
        [summary[str(k)]["f1"] for k in ks],
        "o-",
        label="F1",
        color=PALETTE["neutral"],
        lw=2,
        ms=8,
    )
    ax.plot(
        ks,
        [summary[str(k)]["recall"] for k in ks],
        "s--",
        label="Recall",
        color=PALETTE["accent"],
        lw=2,
        ms=8,
    )
    ax.set_xlabel("Number of copies (k)")
    ax.set_ylabel("Score")
    ax.set_title("(b) Copies ablation")
    ax.set_ylim(0, 1.1)
    ax.set_xticks(ks)
    ax.legend()


def plot_comparison(ax, antidote: dict, spectral: dict) -> None:
    methods = ["Antidote", "Spectral"]
    f1s = [antidote["f1"], spectral["f1"]]
    bars = ax.bar(
        methods,
        f1s,
        color=[PALETTE["antidote"], PALETTE["spectral"]],
        edgecolor="black",
        lw=1.2,
    )
    ax.set_ylabel("Detection F1")
    ax.set_title("(c) Defense comparison")
    ax.set_ylim(0, 1.1)
    for bar, val in zip(bars, f1s):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            fontweight="bold",
        )


def plot_mi(ax, rows: list[dict]) -> None:
    models = [r["model"] for r in rows]
    aucs = [r["mi_auc"] for r in rows]
    colors = [PALETTE["clean"], PALETTE["poisoned"], PALETTE["defended"]]
    bars = ax.bar(models, aucs, color=colors, edgecolor="black", lw=1.2)
    ax.axhline(0.5, color="gray", ls="--", label="Random")
    ax.set_ylabel("MI attack AUC")
    ax.set_title("(d) Membership inference")
    ax.set_ylim(0, 1.0)
    ax.legend()
    for bar, val in zip(bars, aucs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            fontweight="bold",
        )


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    results = Path(cfg["output"]["results_dir"])
    figures = Path(cfg["output"]["figures_dir"])

    with open(results / "table2_threshold_ablation.json") as f:
        threshold_summary = json.load(f)
    with open(results / "table3_copies_ablation.json") as f:
        copies_summary = json.load(f)
    with open(results / "table1_detection.json") as f:
        detection = json.load(f)
    with open(results / "table4_end_to_end.json") as f:
        e2e = json.load(f)

    # --- Four-panel ablations figure --------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plot_threshold(axes[0, 0], threshold_summary)
    plot_copies(axes[0, 1], copies_summary)
    plot_comparison(axes[1, 0], detection["antidote"], detection["spectral_signatures"])
    plot_mi(axes[1, 1], e2e["results"])
    fig.tight_layout()
    fig.savefig(figures / "ablations.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {figures / 'ablations.png'}")

    # --- Hero: loss distributions ------------------------------------------
    losses = np.load(results / "table4_losses.npz")
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = 50
    ax.hist(
        losses["clean_targets"],
        bins=bins,
        alpha=0.55,
        label="Clean (targets)",
        color=PALETTE["clean"],
        density=True,
    )
    ax.hist(
        losses["poisoned_targets"],
        bins=bins,
        alpha=0.55,
        label="Poisoned (targets)",
        color=PALETTE["poisoned"],
        density=True,
    )
    ax.hist(
        losses["defended_targets"],
        bins=bins,
        alpha=0.55,
        label="Defended (targets)",
        color=PALETTE["defended"],
        density=True,
    )
    ax.set_xlabel("Loss on target samples")
    ax.set_ylabel("Density")
    ax.set_title("Target loss distributions across training regimes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "loss_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {figures / 'loss_distributions.png'}")


if __name__ == "__main__":
    main()
