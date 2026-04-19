"""Shared helpers for experiment scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Load experiment config; defaults to configs/default.yaml."""
    if path is None:
        path = REPO_ROOT / "configs" / "default.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def ensure_dirs(cfg: Dict[str, Any]) -> None:
    Path(cfg["output"]["results_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["output"]["figures_dir"]).mkdir(parents=True, exist_ok=True)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"  wrote {path}")


def save_csv_rows(rows: list[dict], path: str | Path) -> None:
    """Minimal CSV writer — avoids pandas dependency for the common case."""
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path}")
