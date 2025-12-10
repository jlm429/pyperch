import torch
import numpy as np

from pyperch import Perch
from pyperch.models import SimpleMLP
from pyperch.core.metrics import Accuracy
from pyperch.utils.seed import set_seed


def make_perch():
    return (
        Perch()
        .model(SimpleMLP, input_dim=10, hidden=[8], output_dim=2)
        .optimizer("rhc", step_size=0.05)
        .metrics(Accuracy())
    )


def test_rhc_reproducible_training():
    # Create deterministic dataset
    set_seed(0)
    X = torch.randn(100, 10)
    y = (X[:, :3].sum(dim=1) > 0).long()

    # Run 1
    set_seed(123)
    perch1 = make_perch().data(X, y, batch_size=32, valid_split=0.2, stratify=False)
    trainer1, hist1 = perch1.train(
        max_epochs=6,
        seed=123,
        optimizer_mode="per_epoch",
    )

    # Run 2
    set_seed(123)
    perch2 = make_perch().data(X, y, batch_size=32, valid_split=0.2, stratify=False)
    trainer2, hist2 = perch2.train(
        max_epochs=6,
        seed=123,
        optimizer_mode="per_epoch",
    )

    assert hist1["train_loss"] == hist2["train_loss"]
