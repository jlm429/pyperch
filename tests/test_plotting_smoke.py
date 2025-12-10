import matplotlib

matplotlib.use("Agg")  # headless backend for CI

from pyperch.utils import plot_losses, plot_metrics


def test_plot_functions_smoke():
    history = {
        "epoch": [0, 1, 2],
        "train_loss": [1.0, 0.8, 0.6],
        "valid_loss": [1.1, 0.9, 0.7],
        "train_metrics": {"accuracy": [0.5, 0.6, 0.7]},
        "valid_metrics": {"accuracy": [0.4, 0.5, 0.6]},
    }

    # Smoke test: ensure plotting functions run without error
    plot_losses(history)
    plot_metrics(history)
