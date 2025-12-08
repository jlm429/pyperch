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


# --------------------------------------------------------
# Generic metric plotting (F1, accuracy, R2, etc.)
# --------------------------------------------------------
def plot_metrics(history: Dict) -> None:
    """Plot ALL metrics for train + valid on one figure.

    Automatically handles:
      - accuracy
      - f1
      - mse
      - r2
      - any new metrics added to history
    """
    epochs: List[int] = history.get("epoch", [])

    train_metrics = history.get("train_metrics", {}) or {}
    valid_metrics = history.get("valid_metrics", {}) or {}

    # Union of all metric names
    metric_names = set(train_metrics.keys()) | set(valid_metrics.keys())

    if not metric_names:
        print("No metrics found in history.")
        return

    plt.figure(figsize=(7, 4))

    for name in sorted(metric_names):
        train_vals = train_metrics.get(name)
        valid_vals = valid_metrics.get(name)

        if train_vals:
            plt.plot(epochs, train_vals, label=f"train_{name}")
        if valid_vals:
            plt.plot(epochs, valid_vals, label=f"valid_{name}")

    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.title("Train & Validation Metrics")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
