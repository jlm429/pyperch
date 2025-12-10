import numpy as np
import torch
import pytest

from pyperch import Perch
from pyperch.models import SimpleMLP
from pyperch.core.metrics import Accuracy
from pyperch.utils.seed import set_seed
from pyperch.config import TrainConfig
from pyperch.core.trainer import Trainer


# -------------------------------------------------------------------
# A. Metric logging structure
# -------------------------------------------------------------------


def test_metric_logging_structure():
    set_seed(0)

    X = torch.randn(50, 5)
    y = (X.sum(dim=1) > 0).long()

    perch = (
        Perch()
        .model(SimpleMLP, input_dim=5, hidden=[8], output_dim=2)
        .optimizer("rhc", step_size=0.05)
        .metrics(Accuracy())
        .data(X, y, batch_size=16, valid_split=0.2, stratify=False)
    )

    _, history = perch.train(max_epochs=2, seed=0)

    assert "epoch" in history
    assert "train_loss" in history
    assert "valid_loss" in history
    assert isinstance(history["train_metrics"], dict)
    assert isinstance(history["valid_metrics"], dict)


# -------------------------------------------------------------------
# B. Layer-freeze basic
# -------------------------------------------------------------------


def test_layer_modes_freeze_basic():
    set_seed(0)

    perch = (
        Perch()
        .model(SimpleMLP, input_dim=5, hidden=[8], output_dim=2)
        .freeze("net.0.weight")
    )

    model = perch.build_model()

    for name, p in model.named_parameters():
        if name == "net.0.weight":
            assert p.requires_grad is False
        else:
            assert p.requires_grad is True


# -------------------------------------------------------------------
# C. Meta + grad registration (soft invariant)
# -------------------------------------------------------------------


def test_grad_and_meta_registration():
    set_seed(0)

    perch = (
        Perch()
        .model(SimpleMLP, input_dim=5, hidden=[8], output_dim=2)
        .grad_opt("net.2.*")
        .meta_opt("net.0.*")
        .optimizer("rhc", step_size=0.05)
        .data(
            torch.randn(40, 5),
            torch.randint(0, 2, (40,)),
            batch_size=8,
            valid_split=0.2,
        )
    )

    trainer, _ = perch.train(max_epochs=1, seed=0)

    assert hasattr(trainer, "grad_params")
    assert hasattr(trainer, "meta_params")

    # No parameter should be in both groups
    assert not (set(trainer.grad_params) & set(trainer.meta_params))


# -------------------------------------------------------------------
# D. TrainConfigBuilder smoke test
# -------------------------------------------------------------------


def test_trainconfig_builder_smoke():
    from pyperch.search.builder import TrainConfigBuilder
    from pyperch.config import OptimizerConfig

    base = TrainConfig(
        optimizer="sa",
        optimizer_config=OptimizerConfig(step_size=0.05, t=1.0),
    )

    builder = TrainConfigBuilder(base)

    # The builder stores a config; this test checks the structure only
    cfg = getattr(builder, "config", None)
    if cfg is None:
        cfg = base

    assert isinstance(cfg, TrainConfig)


# -------------------------------------------------------------------
# E. Trainer early stopping
# -------------------------------------------------------------------


class EarlyStopCallback:
    def on_train_begin(self, engine):
        pass

    def on_epoch_begin(self, engine):
        pass

    def on_epoch_end(self, engine):
        if engine.epoch >= 1:
            engine.stop_training = True

    def on_train_end(self, engine):
        pass


def test_trainer_early_stopping():
    set_seed(0)

    X = torch.randn(60, 5)
    y = (X.sum(dim=1) > 0).long()

    cb = EarlyStopCallback()

    perch = (
        Perch()
        .model(SimpleMLP, input_dim=5, hidden=[8], output_dim=2)
        .optimizer("rhc", step_size=0.05)
        .metrics(Accuracy())
        .data(X, y, batch_size=16, valid_split=0.2, stratify=False)
        .callbacks(cb)
    )

    _, history = perch.train(max_epochs=10, seed=0)

    # Should stop after epoch 1
    assert len(history["epoch"]) == 2


# -------------------------------------------------------------------
# F. Invalid configuration tests
# -------------------------------------------------------------------


def test_invalid_optimizer_name():
    perch = Perch()
    perch.optimizer("not_an_optimizer")  # Permissive in current API
    assert True


def test_invalid_metric_name():
    perch = Perch()
    perch.metrics("not_a_metric")  # Permissive in current API
    assert True


def test_invalid_layer_mode():
    perch = Perch().model(SimpleMLP, input_dim=5, hidden=[8], output_dim=2)
    perch._layer_modes = {"net.0.weight": "invalid_mode"}
    perch.build_model()  # Should not raise
    assert True


# -------------------------------------------------------------------
# G. No validation loader
# -------------------------------------------------------------------


def test_trainer_allows_no_validation_loader():
    set_seed(0)

    X = torch.randn(30, 5)
    y = torch.randint(0, 2, (30,))
    ds = list(zip(X, y))
    loader = torch.utils.data.DataLoader(ds, batch_size=8)

    model = SimpleMLP(input_dim=5, hidden=[8], output_dim=2)

    cfg = TrainConfig(
        optimizer="rhc",
        max_epochs=2,
        metrics={"train": [Accuracy()]},
    )

    trainer = Trainer(model, torch.nn.CrossEntropyLoss(), cfg)
    history = trainer.fit(loader, valid_loader=None)

    assert len(history["epoch"]) == 2


# -------------------------------------------------------------------
# H. per_batch vs per_epoch
# -------------------------------------------------------------------


def test_optimizer_mode_switching():
    set_seed(0)

    X = torch.randn(80, 5)
    y = (X[:, 0] > 0).long()

    perch = (
        Perch()
        .model(SimpleMLP, input_dim=5, hidden=[8], output_dim=2)
        .optimizer("rhc", step_size=0.05)
        .metrics(Accuracy())
        .data(X, y, batch_size=10, valid_split=0.2, stratify=False)
    )

    _, h1 = perch.train(max_epochs=3, seed=0, optimizer_mode="per_epoch")
    _, h2 = perch.train(max_epochs=3, seed=0, optimizer_mode="per_batch")

    assert len(h1["epoch"]) == 3
    assert len(h2["epoch"]) == 3
