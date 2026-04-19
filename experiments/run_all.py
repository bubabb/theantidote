"""Run every experiment in order.

Usage:
    python experiments/run_all.py

Each table script is a standalone entry point; this script simply invokes them
sequentially so reviewers can regenerate every paper result with one command.
"""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

SCRIPTS = [
    "table1_detection",
    "table2_threshold_ablation",
    "table3_copies_ablation",
    "table4_end_to_end",
    "make_figures",
]


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    for name in SCRIPTS:
        print(f"\n{'=' * 70}\n  {name}\n{'=' * 70}")
        t0 = time.time()
        mod = importlib.import_module(name)
        mod.main()
        print(f"  {name} finished in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
