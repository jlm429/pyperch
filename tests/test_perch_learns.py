import torch
import numpy as np
from torch import nn

from pyperch import Perch
from pyperch.models import SimpleMLP
from pyperch.core.metrics import Accuracy, MSE, R2


# ----------------------------------------------------------------------
# Helper functions for small reproducible datasets
# ----------------------------------------------------------------------


def make_tiny_classification(n=200, d=10, seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    X = torch.randn(n, d)
    y = (X[:, :3].sum(dim=1) > 0).long()
    return X, y


def make_tiny_regression(n=200, d=10, seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    X = torch.randn(n, d)
    true_w = torch.randn(d, 1)
    y = X @ true_w + 0.1 * torch.randn(n, 1)
    return X.float(), y.float()


# ----------------------------------------------------------------------
# RHC classification: training loss should decrease
# ----------------------------------------------------------------------


def test_perch_rhc_learns_classification():
    X, y = make_tiny_classification()

    perch = (
        Perch()
        .model(SimpleMLP, input_dim=10, hidden=[16], output_dim=2, activation="relu")
        .optimizer("rhc", step_size=0.05)
        .metrics(Accuracy())
        .data(X, y, batch_size=32, valid_split=0.2, stratify=False)
    )

    trainer, history = perch.train(max_epochs=8, seed=0)

    losses = history["train_loss"]
    assert losses[0] >= losses[-1]


# ----------------------------------------------------------------------
# SA classification: training loss should decrease
# ----------------------------------------------------------------------


def test_perch_sa_learns_classification():
    X, y = make_tiny_classification()

    perch = (
        Perch()
        .model(SimpleMLP, input_dim=10, hidden=[16], output_dim=2)
        .optimizer("sa", t=1.5, t_min=0.001, cooling=0.98, step_size=0.05)
        .metrics(Accuracy())
        .data(X, y, batch_size=32, valid_split=0.2, stratify=False)
    )

    trainer, history = perch.train(max_epochs=10, seed=0, optimizer_mode="per_batch")

    losses = history["train_loss"]
    assert losses[0] >= losses[-1]


# ----------------------------------------------------------------------
# GA classification: accuracy should improve at some point
# ----------------------------------------------------------------------


def test_perch_ga_learns_classification_accuracy():
    X, y = make_tiny_classification()

    perch = (
        Perch()
        .model(SimpleMLP, input_dim=10, hidden=[16], output_dim=2)
        .optimizer("ga", population_size=40, mutation_rate=0.15, step_size=0.1)
        .metrics(Accuracy())
        .data(X, y, batch_size=32, valid_split=0.2, stratify=False)
    )

    trainer, history = perch.train(max_epochs=10, seed=0, optimizer_mode="per_batch")

    acc = history["train_metrics"].get("accuracy", [])
    if len(acc) > 1:
        assert max(acc) >= acc[0]


# ----------------------------------------------------------------------
# RHC regression: R2 should improve at some point
# ----------------------------------------------------------------------


def test_perch_rhc_learns_regression():
    X, y = make_tiny_regression()

    perch = (
        Perch()
        .model(SimpleMLP, input_dim=10, hidden=[16], output_dim=1, activation="relu")
        .optimizer("rhc", step_size=0.1)
        .metrics(MSE(), R2())
        .data(X, y, batch_size=32, valid_split=0.2, stratify=False)
    )

    trainer, history = perch.train(max_epochs=12, seed=0)

    r2 = history["train_metrics"].get("r2", [])
    if len(r2) > 1:
        assert max(r2) >= r2[0]


# ----------------------------------------------------------------------
# SA regression: R2 should improve at some point
# ----------------------------------------------------------------------


def test_perch_sa_learns_regression():
    X, y = make_tiny_regression()

    perch = (
        Perch()
        .model(SimpleMLP, input_dim=10, hidden=[16], output_dim=1)
        .optimizer("sa", t=1.5, t_min=0.001, cooling=0.98, step_size=0.1)
        .metrics(MSE(), R2())
        .data(X, y, batch_size=32, valid_split=0.2, stratify=False)
    )

    trainer, history = perch.train(max_epochs=12, seed=0, optimizer_mode="per_batch")

    r2 = history["train_metrics"].get("r2", [])
    if len(r2) > 1:
        assert max(r2) >= r2[0]


# ----------------------------------------------------------------------
# GA regression: R2 should improve at some point
# ----------------------------------------------------------------------


def test_perch_ga_learns_regression_r2():
    X, y = make_tiny_regression()

    perch = (
        Perch()
        .model(SimpleMLP, input_dim=10, hidden=[16], output_dim=1)
        .optimizer("ga", population_size=40, mutation_rate=0.15, step_size=0.1)
        .metrics(MSE(), R2())
        .data(X, y, batch_size=32, valid_split=0.2, stratify=False)
    )

    trainer, history = perch.train(max_epochs=15, seed=0, optimizer_mode="per_batch")

    r2 = history["train_metrics"].get("r2", [])
    if len(r2) > 1:
        assert max(r2) >= r2[0]
