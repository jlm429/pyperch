import torch
from torch import nn

from pyperch.optim import GA


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


def test_ga_classification_runs_and_tracks_counters():
    torch.manual_seed(42)

    X, y = make_classification_data()
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

    criterion = nn.CrossEntropyLoss()
    optimizer = GA(
        model.parameters(),
        population_size=8,
        mutation_rate=0.1,
        step_size=0.05,
    )

    def closure():
        optimizer.zero_grad()
        return criterion(model(X), y)

    initial_loss = closure().item()

    for _ in range(10):
        optimizer.step(closure)

    final_loss = closure().item()

    assert torch.isfinite(torch.tensor(final_loss))
    assert optimizer.function_evals > 0
    assert optimizer.proposed_steps > 0
    assert optimizer.accepted_steps > 0
    assert optimizer.rejected_steps >= 0
    assert optimizer.best_loss is not None
    assert optimizer.best_loss <= initial_loss


def test_ga_regression_runs_and_tracks_best_loss():
    torch.manual_seed(42)

    X, y = make_regression_data()
    model = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 1))

    criterion = nn.MSELoss()
    optimizer = GA(
        model.parameters(),
        population_size=8,
        mutation_rate=0.1,
        step_size=0.05,
    )

    def closure():
        optimizer.zero_grad()
        return criterion(model(X), y)

    initial_loss = closure().item()

    for _ in range(10):
        optimizer.step(closure)

    assert optimizer.best_loss is not None
    assert optimizer.best_loss <= initial_loss
    assert optimizer.function_evals > 0


def test_ga_respects_frozen_parameters():
    torch.manual_seed(42)

    X, y = make_classification_data()
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

    for param in model[-1].parameters():
        param.requires_grad = False

    frozen_before = [p.detach().clone() for p in model[-1].parameters()]

    criterion = nn.CrossEntropyLoss()
    optimizer = GA(
        model.parameters(),
        population_size=8,
        mutation_rate=0.1,
        step_size=0.05,
    )

    def closure():
        optimizer.zero_grad()
        return criterion(model(X), y)

    for _ in range(5):
        optimizer.step(closure)

    frozen_after = [p.detach().clone() for p in model[-1].parameters()]

    for before, after in zip(frozen_before, frozen_after):
        assert torch.equal(before, after)


def test_ga_reset_counters():
    torch.manual_seed(42)

    X, y = make_classification_data()
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

    criterion = nn.CrossEntropyLoss()
    optimizer = GA(
        model.parameters(),
        population_size=8,
        mutation_rate=0.1,
        step_size=0.05,
    )

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
