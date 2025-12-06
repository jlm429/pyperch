from __future__ import annotations

from typing import Dict, List

import matplotlib.pyplot as plt


def plot_losses(history: Dict) -> None:
    epochs: List[int] = history.get("epoch", [])
    train_loss = history.get("train_loss", [])
    valid_loss = history.get("valid_loss", [])

    plt.figure()
    if train_loss:
        plt.plot(epochs, train_loss, label="train_loss")
    if valid_loss:
        plt.plot(epochs, valid_loss, label="valid_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    
    
def plot_accuracy(history: Dict) -> None:
    """Plot train + validation accuracy on a single figure."""
    epochs: List[int] = history.get("epoch", [])
    train_metrics = history.get("train_metrics", {})
    valid_metrics = history.get("valid_metrics", {})

    train_acc = train_metrics.get("accuracy", [])
    valid_acc = valid_metrics.get("accuracy", [])

    plt.figure()
    if train_acc:
        plt.plot(epochs, train_acc, label="Train Accuracy")
    if valid_acc:
        plt.plot(epochs, valid_acc, label="Validation Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def plot_metrics(history: Dict, split: str = "valid") -> None:
    key = f"{split}_metrics"
    epochs: List[int] = history.get("epoch", [])
    metrics = history.get(key, {})
    if not metrics:
        return

    plt.figure()
    for name, values in metrics.items():
        plt.plot(epochs, values, label=name)
    plt.xlabel("epoch")
    plt.ylabel("metric value")
    plt.legend()
    plt.tight_layout()
