# Antidote

> **First targeted defense against Truth Serum privacy attacks.**
> Detect near-duplicate label-flipping poisons with pretrained ImageNet features — before training corruption occurs.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Paper](https://img.shields.io/badge/paper-PDF-red.svg)](paper/antidote_paper.pdf)

---

## TL;DR

The **Truth Serum** attack (Tramèr et al., CCS 2022) amplifies membership inference by poisoning training data with mislabeled copies of target samples. The original authors explicitly left defenses as an open problem.

**Antidote** exploits the attack's unavoidable signature — *near-duplicate samples with inconsistent labels* — using pretrained ImageNet features that are immune to the poisoning we're trying to detect.

| Method | Precision | Recall | F1 |
|---|---|---|---|
| **Antidote (ours)** | 0.889 | **1.000** | **0.941** |
| Spectral Signatures | 0.009 | 0.012 | 0.010 |

**End-to-end:** MI AUC reduced from 0.546 → 0.520 (**52.9% reduction** in privacy amplification), test accuracy only 7.1% below clean baseline.

---

## Quickstart

```bash
git clone https://github.com/bubabb/theantidote.git
cd theantidote
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Reproduce every paper table
python experiments/run_all.py
```

Outputs land in `results/` as CSVs and figures.

## Repository layout

```
theantidote/
├── src/antidote/           # core library
│   ├── defense.py          # Antidote detector (near-duplicate + label mismatch)
│   ├── attack.py           # Truth Serum poisoning
│   ├── models.py           # ResNet-18 (CIFAR-adapted)
│   ├── data.py             # CIFAR-10 loading + poison injection
│   ├── training.py         # training loop + membership inference scoring
│   └── baselines/
│       └── spectral.py     # Spectral Signatures baseline
├── experiments/            # runnable scripts per paper table
├── configs/                # YAML configs for attack + defense
├── results/                # generated CSVs and figures (committed)
├── paper/                  # paper PDF + BibTeX
├── notebooks/              # exploratory analysis + figure generation
└── tests/                  # unit tests
```

## Method

Two phases:

1. **Feature extraction** — embed all training samples with ImageNet-pretrained ResNet-18 (never exposed to the potentially poisoned data). Breaks the circular dependency of using a corrupted model to detect corruption.
2. **Near-duplicate detection** — for each sample, find neighbors with cosine similarity > τ (default 0.99). Flag as poison if any near-duplicate has a mismatched label OR the duplicate cluster has ≥3 members.

See `src/antidote/defense.py` for the full implementation.

## Reproduction

Each script below regenerates a specific table from the paper:

| Script | Paper result |
|---|---|
| `experiments/table1_detection.py` | Detection F1 (Antidote vs Spectral Signatures) |
| `experiments/table2_threshold_ablation.py` | Similarity threshold sweep |
| `experiments/table3_copies_ablation.py` | Number of poison copies sweep |
| `experiments/table4_end_to_end.py` | MI AUC: clean vs poisoned vs defended |

All experiments use CIFAR-10 and a ResNet-18 (CIFAR-adapted). A single A100 GPU reproduces the full set in roughly two hours.

## Citation

```bibtex
@misc{vasconcelos2025antidote,
  title  = {Antidote: First Targeted Defenses Against Truth Serum Privacy Attacks},
  author = {Vasconcelos, Bruna},
  year   = {2025},
  note   = {CS 6958: Machine Learning Security, University of Utah},
  url    = {https://github.com/bubabb/theantidote}
}
```

## References

- [1] Tramèr et al. *Truth Serum: Poisoning Machine Learning Models to Reveal Their Secrets.* ACM CCS 2022.
- [2] Shokri et al. *Membership Inference Attacks Against Machine Learning Models.* IEEE S&P 2017.
- [3] Tran et al. *Spectral Signatures in Backdoor Attacks.* NeurIPS 2018.

## License

MIT — see [LICENSE](LICENSE).
