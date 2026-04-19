"""Training loop and accuracy evaluation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim


def train_model(
    model: nn.Module,
    trainloader: torch.utils.data.DataLoader,
    epochs: int = 25,
    lr: float = 0.1,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    device: torch.device | str | None = None,
    log_every: int = 5,
) -> nn.Module:
    """SGD + cosine annealing training loop.

    Defaults match the paper: lr=0.1, momentum=0.9, wd=5e-4, 25 epochs.
    """
    device = torch.device(
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for epoch in range(epochs):
        for inputs, targets in trainloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
        scheduler.step()
        if log_every and (epoch + 1) % log_every == 0:
            print(f"  epoch {epoch + 1}/{epochs}")
    return model


@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module,
    testloader: torch.utils.data.DataLoader,
    device: torch.device | str | None = None,
) -> float:
    device = torch.device(
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = model.to(device).eval()
    correct, total = 0, 0
    for inputs, targets in testloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        _, predicted = model(inputs).max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return correct / total
