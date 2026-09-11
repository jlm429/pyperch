import pytest
import torch
from torch import nn

from pyperch.optim import SA


def make_classification_data(n=64, d=4):
    torch.manual_seed(42)
    X = torch.randn(n, d)
    y = (X.sum(dim=1) > 0).long()
    return X, y


def make_regression_data(n=64, d=3):
    torch.manual_seed(42)
    X = torch.randn(n, d)
    y = X.sum(dim=1, keepdim=True)
    return X, y


def test_sa_classification_runs_and_tracks_counters():
    torch.manual_seed(42)

    X, y = make_classification_data()
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

    criterion = nn.CrossEntropyLoss()
    optimizer = SA(model.parameters(), step_size=0.05, temperature=1.0, cooling=0.95)

    def closure():
        optimizer.zero_grad()
        return criterion(model(X), y)

    initial_loss = closure().item()

    for _ in range(50):
        optimizer.step(closure)

    final_loss = closure().item()

    assert torch.isfinite(torch.tensor(final_loss))
    assert optimizer.function_evals > 0
    assert optimizer.proposed_steps > 0
    assert (
        optimizer.accepted_steps + optimizer.rejected_steps == optimizer.proposed_steps
    )
    assert optimizer.best_loss is not None
    assert optimizer.best_loss <= initial_loss


def test_sa_regression_runs_and_tracks_best_loss():
    torch.manual_seed(42)

    X, y = make_regression_data()
    model = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 1))

    criterion = nn.MSELoss()
    optimizer = SA(model.parameters(), step_size=0.05, temperature=1.0, cooling=0.95)

    def closure():
        optimizer.zero_grad()
        return criterion(model(X), y)

    initial_loss = closure().item()

    for _ in range(50):
        optimizer.step(closure)

    assert optimizer.best_loss is not None
    assert optimizer.best_loss <= initial_loss
    assert optimizer.function_evals > 0


def test_sa_respects_frozen_parameters():
    torch.manual_seed(42)

    X, y = make_classification_data()
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

    for param in model[0].parameters():
        param.requires_grad = False

    frozen_before = [p.detach().clone() for p in model[0].parameters()]

    criterion = nn.CrossEntropyLoss()
    optimizer = SA(model.parameters(), step_size=0.05, temperature=1.0, cooling=0.95)

    def closure():
        optimizer.zero_grad()
        return criterion(model(X), y)

    for _ in range(20):
        optimizer.step(closure)

    frozen_after = [p.detach().clone() for p in model[0].parameters()]

    for before, after in zip(frozen_before, frozen_after):
        assert torch.equal(before, after)


def test_sa_reset_counters():
    torch.manual_seed(42)

    X, y = make_classification_data()
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

    criterion = nn.CrossEntropyLoss()
    optimizer = SA(model.parameters(), step_size=0.05, temperature=1.0, cooling=0.95)

    def closure():
        optimizer.zero_grad()
        return criterion(model(X), y)

    optimizer.step(closure)
    optimizer.reset_counters()

    assert optimizer.function_evals == 0
    assert optimizer.proposed_steps == 0
    assert optimizer.accepted_steps == 0
    assert optimizer.rejected_steps == 0
    assert optimizer.best_loss is None


def test_sa_reset_counters_starts_fresh_run_and_temperature_schedule():
    parameter = nn.Parameter(torch.tensor([0.0]))
    optimizer = SA(
        [parameter],
        step_size=0.1,
        temperature=2.0,
        min_temperature=0.1,
        cooling=0.5,
        random_state=7,
    )

    def closure():
        return parameter.square().sum()

    optimizer.step(closure)
    optimizer.step(closure)
    assert optimizer.temperature == 1.0

    with torch.no_grad():
        parameter.fill_(5)
    optimizer.reset_counters()
    fresh_loss = optimizer.step(closure)

    assert fresh_loss.item() == 25
    assert optimizer.best_loss == 25
    assert optimizer.function_evals == 1
    assert optimizer.proposed_steps == 0
    assert optimizer.accepted_steps == 0
    assert optimizer.rejected_steps == 0
    assert optimizer.temperature == 2.0

    with torch.no_grad():
        parameter.fill_(7)
    optimizer.restore_best()

    assert parameter.item() == 5


def test_sa_restore_best_synchronizes_continued_optimization():
    parameter = nn.Parameter(torch.tensor([0.0]))
    optimizer = SA(
        [parameter],
        step_size=2.0,
        temperature=1000.0,
        min_temperature=1e-12,
        cooling=1.0,
        random_state=0,
    )

    def closure():
        return parameter.square().sum()

    optimizer.step(closure)
    optimizer.step(closure)
    optimizer.temperature = 1e-12
    optimizer.restore_best()
    optimizer.step(closure)

    assert parameter.item() == 0
    assert optimizer.accepted_steps == 1
    assert optimizer.rejected_steps == 1


@pytest.mark.parametrize(
    "options",
    [
        {"temperature": 0.0},
        {"temperature": float("inf")},
        {"temperature": float("nan")},
        {"min_temperature": 0.0},
        {"min_temperature": float("inf")},
        {"min_temperature": float("nan")},
        {"temperature": 1.0, "min_temperature": 1.1},
        {"cooling": 0.0},
        {"cooling": 1.1},
        {"cooling": float("inf")},
        {"cooling": float("nan")},
    ],
)
def test_sa_rejects_invalid_temperature_configuration(options):
    parameter = nn.Parameter(torch.zeros(1))

    with pytest.raises(ValueError):
        SA([parameter], **options)
