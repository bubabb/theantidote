"""Antidote: targeted defense against Truth Serum privacy attacks."""

from antidote.attack import TruthSerumAttack
from antidote.data import PoisonedDataset, cifar10_datasets
from antidote.defense import AntidoteDefense
from antidote.baselines.spectral import SpectralSignaturesDefense
from antidote.models import create_resnet18_cifar
from antidote.training import train_model, evaluate_accuracy
from antidote.membership_inference import (
    compute_mi_truth_serum,
    compute_mi_standard,
    get_losses,
)
from antidote.metrics import evaluate_detection

__version__ = "0.1.0"

__all__ = [
    "TruthSerumAttack",
    "PoisonedDataset",
    "cifar10_datasets",
    "AntidoteDefense",
    "SpectralSignaturesDefense",
    "create_resnet18_cifar",
    "train_model",
    "evaluate_accuracy",
    "compute_mi_truth_serum",
    "compute_mi_standard",
    "get_losses",
    "evaluate_detection",
]
