import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pyperch import Trainer
from pyperch.config import TrainConfig, OptimizerConfig


def tiny_model_3layer():
    return nn.Sequential(
        nn.Linear(4, 4),  # 0
        nn.ReLU(),  # 1
        nn.Linear(4, 4),  # 2
        nn.ReLU(),  # 3
        nn.Linear(4, 2),  # 4
    )


def tiny_model_2layer():
    return nn.Sequential(
        nn.Linear(4, 4),  # 0
        nn.ReLU(),
        nn.Linear(4, 2),  # 2
    )


# ----------------------------------------------------------------------
# Utility: Run RHC multiple times to reduce test flake
# ----------------------------------------------------------------------
def rhc_run_until_change(create_model_fn, cfg, loader, attempts=5):
    """
    RHC is stochastic, so sometimes no update happens in a single short run.
    This helper retries several times and returns True if a change occurs.
    """
    for _ in range(attempts):
        model = create_model_fn()
        before = [p.detach().clone() for p in model.parameters()]

        trainer = Trainer(model=model, loss_fn=nn.CrossEntropyLoss(), config=cfg)
        trainer.fit(loader, loader)

        after = [p.detach() for p in model.parameters()]

        if any(not torch.equal(b, a) for b, a in zip(before, after)):
            return True  # change detected

    return False  # no change after multiple attempts


# ----------------------------------------------------------------------
# Test 1: Freeze first layer
# ----------------------------------------------------------------------
def test_freeze_first_layer_trainer():

    model = tiny_model_2layer()
    X = torch.randn(32, 4)
    y = (X.sum(dim=1) > 0).long()
    loader = DataLoader(TensorDataset(X, y), batch_size=8)

    cfg = TrainConfig(
        optimizer="rhc",
        optimizer_config=OptimizerConfig(step_size=0.1),
        max_epochs=5,
        layer_modes={
            "0.weight": "freeze",
            "0.bias": "freeze",
        },
    )

    before = model[0].weight.detach().clone()

    trainer = Trainer(model=model, loss_fn=nn.CrossEntropyLoss(), config=cfg)
    trainer.fit(loader, loader)

    after = model[0].weight.detach()

    assert torch.equal(before, after)


# ----------------------------------------------------------------------
# Test 2: Freeze middle layer
# ----------------------------------------------------------------------
def test_freeze_middle_layer():

    X = torch.randn(32, 4)
    y = (X.sum(dim=1) > 0).long()
    loader = DataLoader(TensorDataset(X, y), batch_size=8)

    cfg = TrainConfig(
        optimizer="rhc",
        optimizer_config=OptimizerConfig(step_size=0.15),
        max_epochs=5,
        layer_modes={
            "2.weight": "freeze",
            "2.bias": "freeze",
        },
    )

    # Test only the trainable layer:
    def make_model():
        return tiny_model_3layer()

    changed = rhc_run_until_change(make_model, cfg, loader)

    assert changed, "Trainable parameters did not change after multiple RHC attempts"


# ----------------------------------------------------------------------
# Test 3: Freeze output layer only
# ----------------------------------------------------------------------
def test_freeze_output_layer():

    model = tiny_model_2layer()
    X = torch.randn(32, 4)
    y = (X.sum(dim=1) > 0).long()
    loader = DataLoader(TensorDataset(X, y), batch_size=8)

    cfg = TrainConfig(
        optimizer="rhc",
        optimizer_config=OptimizerConfig(step_size=0.1),
        max_epochs=5,
        layer_modes={
            "2.weight": "freeze",
            "2.bias": "freeze",
        },
    )

    before_frozen = model[2].weight.detach().clone()
    before_trainable = model[0].weight.detach().clone()

    trainer = Trainer(model=model, loss_fn=nn.CrossEntropyLoss(), config=cfg)
    trainer.fit(loader, loader)

    after_frozen = model[2].weight.detach()
    after_trainable = model[0].weight.detach()

    assert torch.equal(before_frozen, after_frozen)
    assert not torch.equal(before_trainable, after_trainable)


# ----------------------------------------------------------------------
# Test 4: No freezing -> all parameters should update
# ----------------------------------------------------------------------
def test_no_freezing_all_trainable():

    X = torch.randn(32, 4)
    y = (X.sum(dim=1) > 0).long()
    loader = DataLoader(TensorDataset(X, y), batch_size=8)

    cfg = TrainConfig(
        optimizer="rhc",
        optimizer_config=OptimizerConfig(step_size=0.15),
        max_epochs=5,
        layer_modes={},
    )

    changed = rhc_run_until_change(tiny_model_2layer, cfg, loader)
    assert changed, "Expected at least one parameter to change with RHC"


# ----------------------------------------------------------------------
# Test 5: requires_grad correctness
# ----------------------------------------------------------------------
def test_freeze_requires_grad_assignment():

    model = tiny_model_2layer()

    cfg = TrainConfig(
        optimizer="rhc",
        optimizer_config=OptimizerConfig(step_size=0.1),
        layer_modes={
            "0.weight": "freeze",
            "0.bias": "freeze",
        },
    )

    Trainer(model=model, loss_fn=nn.CrossEntropyLoss(), config=cfg)

    assert model[0].weight.requires_grad is False
    assert model[0].bias.requires_grad is False

    # second linear layer still trainable
    assert model[2].weight.requires_grad is True
    assert model[2].bias.requires_grad is True


# ----------------------------------------------------------------------
# Test 6: Mixed freeze, grad, meta
# ----------------------------------------------------------------------
def test_freeze_grad_meta_mix():

    model = tiny_model_3layer()

    cfg = TrainConfig(
        optimizer="rhc",
        optimizer_config=OptimizerConfig(step_size=0.1),
        layer_modes={
            "0.weight": "freeze",
            "0.bias": "freeze",
            "2.weight": "grad",
            "2.bias": "grad",
        },
    )

    Trainer(model=model, loss_fn=nn.CrossEntropyLoss(), config=cfg)

    assert model[0].weight.requires_grad is False
    assert model[2].weight.requires_grad is True
    assert model[4].weight.requires_grad is True
