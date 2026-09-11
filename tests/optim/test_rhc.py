import io

import pytest
import torch
from torch import nn

from pyperch.optim import RHC


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


def make_scalar_optimizer(*, seed, restarts, restart_interval):
    torch.manual_seed(0)
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()

    optimizer = RHC(
        model.parameters(),
        step_size=1.0,
        restarts=restarts,
        restart_interval=restart_interval,
        random_state=seed,
    )
    X = torch.ones(1, 1)
    y = torch.full((1, 1), -100.0)
    criterion = nn.MSELoss()

    def closure():
        return criterion(model(X), y)

    return optimizer, closure


def test_rhc_classification_improves_loss():
    torch.manual_seed(42)

    X, y = make_classification_data()
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

    criterion = nn.CrossEntropyLoss()
    optimizer = RHC(model.parameters(), step_size=0.05)

    def closure():
        optimizer.zero_grad()
        return criterion(model(X), y)

    initial_loss = closure().item()

    for _ in range(50):
        optimizer.step(closure)

    final_loss = closure().item()

    assert final_loss <= initial_loss
    assert optimizer.function_evals > 0
    assert optimizer.proposed_steps > 0
    assert (
        optimizer.accepted_steps + optimizer.rejected_steps == optimizer.proposed_steps
    )
    assert optimizer.best_loss is not None


def test_rhc_regression_improves_loss():
    torch.manual_seed(42)

    X, y = make_regression_data()
    model = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 1))

    criterion = nn.MSELoss()
    optimizer = RHC(model.parameters(), step_size=0.05)

    def closure():
        optimizer.zero_grad()
        return criterion(model(X), y)

    initial_loss = closure().item()

    for _ in range(50):
        optimizer.step(closure)

    final_loss = closure().item()

    assert final_loss <= initial_loss
    assert optimizer.best_loss is not None


def test_rhc_respects_frozen_parameters():
    torch.manual_seed(42)

    X, y = make_classification_data()
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

    for param in model[-1].parameters():
        param.requires_grad = False

    frozen_before = [p.detach().clone() for p in model[-1].parameters()]

    criterion = nn.CrossEntropyLoss()
    optimizer = RHC(model.parameters(), step_size=0.05)

    def closure():
        optimizer.zero_grad()
        return criterion(model(X), y)

    for _ in range(20):
        optimizer.step(closure)

    frozen_after = [p.detach().clone() for p in model[-1].parameters()]

    for before, after in zip(frozen_before, frozen_after):
        assert torch.equal(before, after)


def test_rhc_reset_counters():
    torch.manual_seed(42)

    X, y = make_classification_data()
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

    criterion = nn.CrossEntropyLoss()
    optimizer = RHC(model.parameters(), step_size=0.05)

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


def test_rhc_reset_counters_starts_fresh_run_and_restart_schedule():
    parameter = nn.Parameter(torch.tensor([0.0]))
    optimizer = RHC(
        [parameter],
        step_size=1.0,
        restarts=1,
        restart_interval=1,
        random_state=7,
    )

    def closure():
        return parameter.square().sum()

    optimizer.step(closure)
    optimizer.step(closure)
    optimizer.step(closure)
    assert optimizer.completed_restarts == 1

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
    assert optimizer.completed_restarts == 0

    with torch.no_grad():
        parameter.fill_(7)
    optimizer.restore_best()

    assert parameter.item() == 5


def test_rhc_restore_best_synchronizes_continued_optimization():
    parameter = nn.Parameter(torch.tensor([0.0]))
    optimizer = RHC(
        [parameter],
        step_size=1.0,
        restarts=1,
        restart_interval=1,
        random_state=1,
    )

    def closure():
        return parameter.square().sum()

    optimizer.step(closure)
    optimizer.step(closure)
    optimizer.step(closure)
    optimizer.restore_best()
    optimizer.step(closure)

    assert parameter.item() == 0
    assert optimizer.accepted_steps == 0
    assert optimizer.rejected_steps == 2


def test_rhc_accepted_boundary_proposal_restarts():
    optimizer, closure = make_scalar_optimizer(seed=10, restarts=1, restart_interval=1)

    optimizer.step(closure)
    optimizer.step(closure)

    assert optimizer.accepted_steps == 1
    assert optimizer.rejected_steps == 0
    assert optimizer.completed_restarts == 1


def test_rhc_rejected_boundary_proposal_restarts():
    optimizer, closure = make_scalar_optimizer(seed=3, restarts=1, restart_interval=1)

    optimizer.step(closure)
    optimizer.step(closure)

    assert optimizer.accepted_steps == 0
    assert optimizer.rejected_steps == 1
    assert optimizer.completed_restarts == 1


def _assert_proposal_outcome(optimizer, *, accepted, rejected):
    assert optimizer.proposed_steps == 1
    assert optimizer.accepted_steps == accepted
    assert optimizer.rejected_steps == rejected
    assert optimizer.completed_restarts == 0


def test_rhc_non_boundary_proposals_do_not_restart():
    accepted_optimizer, accepted_closure = make_scalar_optimizer(
        seed=10, restarts=1, restart_interval=2
    )
    rejected_optimizer, rejected_closure = make_scalar_optimizer(
        seed=3, restarts=1, restart_interval=2
    )

    accepted_optimizer.step(accepted_closure)
    accepted_optimizer.step(accepted_closure)
    rejected_optimizer.step(rejected_closure)
    rejected_optimizer.step(rejected_closure)

    _assert_proposal_outcome(accepted_optimizer, accepted=1, rejected=0)
    _assert_proposal_outcome(rejected_optimizer, accepted=0, rejected=1)


def test_rhc_mixed_outcomes_restart_at_proposal_multiples_until_exhausted():
    optimizer, closure = make_scalar_optimizer(seed=13, restarts=2, restart_interval=2)

    optimizer.step(closure)
    restart_proposals = []
    outcomes = []
    while optimizer.proposed_steps < 6:
        previous_restarts = optimizer.completed_restarts
        previous_accepted = optimizer.accepted_steps
        previous_rejected = optimizer.rejected_steps
        optimizer.step(closure)
        if optimizer.completed_restarts > previous_restarts:
            restart_proposals.append(optimizer.proposed_steps)
        if optimizer.accepted_steps > previous_accepted:
            outcomes.append("accepted")
        elif optimizer.rejected_steps > previous_rejected:
            outcomes.append("rejected")

    assert "accepted" in outcomes
    assert "rejected" in outcomes
    assert restart_proposals == [2, 4]
    assert optimizer.completed_restarts == 2
    assert optimizer.proposed_steps == 6
    assert optimizer.accepted_steps + optimizer.rejected_steps == 6


def test_rhc_post_restart_evaluation_does_not_count_as_proposal():
    optimizer, closure = make_scalar_optimizer(seed=10, restarts=1, restart_interval=1)

    optimizer.step(closure)
    optimizer.step(closure)
    counters_after_restart = (
        optimizer.function_evals,
        optimizer.proposed_steps,
        optimizer.accepted_steps,
        optimizer.rejected_steps,
    )
    optimizer.step(closure)

    assert counters_after_restart == (2, 1, 1, 0)
    assert optimizer.function_evals == 3
    assert optimizer.proposed_steps == 1
    assert optimizer.accepted_steps == 1
    assert optimizer.rejected_steps == 0
    assert optimizer.completed_restarts == 1


def _assert_restarts_disabled(optimizer):
    assert optimizer.completed_restarts == 0
    assert optimizer.proposed_steps == 4
    assert optimizer.accepted_steps + optimizer.rejected_steps == 4


def test_rhc_disabled_restart_configurations_remain_disabled():
    no_interval_optimizer, no_interval_closure = make_scalar_optimizer(
        seed=13, restarts=2, restart_interval=None
    )
    no_budget_optimizer, no_budget_closure = make_scalar_optimizer(
        seed=13, restarts=0, restart_interval=1
    )

    for optimizer, closure in (
        (no_interval_optimizer, no_interval_closure),
        (no_budget_optimizer, no_budget_closure),
    ):
        optimizer.step(closure)
        for _ in range(4):
            optimizer.step(closure)

    _assert_restarts_disabled(no_interval_optimizer)
    _assert_restarts_disabled(no_budget_optimizer)


def test_rhc_restart_scale_controls_reset_magnitude_and_preserves_frozen_params():
    def run(scale):
        trainable = nn.Parameter(torch.zeros(4))
        frozen = nn.Parameter(torch.full((4,), 7.0), requires_grad=False)
        optimizer = RHC(
            [trainable, frozen],
            step_size=0.1,
            restarts=1,
            restart_interval=1,
            restart_scale=scale,
            random_state=19,
        )
        optimizer.step(lambda: torch.zeros(()))
        optimizer.step(lambda: torch.zeros(()))
        return trainable.detach().clone(), frozen.detach().clone()

    unit_trainable, unit_frozen = run(1.0)
    scaled_trainable, scaled_frozen = run(0.25)

    assert torch.allclose(scaled_trainable, unit_trainable * 0.25)
    assert torch.equal(unit_frozen, torch.full((4,), 7.0))
    assert torch.equal(scaled_frozen, unit_frozen)


def test_rhc_restart_scale_supports_parameter_groups():
    def run(second_scale):
        first = nn.Parameter(torch.zeros(3))
        second = nn.Parameter(torch.zeros(3))
        optimizer = RHC(
            [
                {"params": [first], "restart_scale": 0.2},
                {"params": [second], "restart_scale": second_scale},
            ],
            step_size=0.1,
            restarts=1,
            restart_interval=1,
            random_state=29,
        )
        optimizer.step(lambda: torch.zeros(()))
        optimizer.step(lambda: torch.zeros(()))
        return first.detach().clone(), second.detach().clone()

    first_small, second_small = run(0.2)
    first_large, second_large = run(0.8)

    assert torch.equal(first_large, first_small)
    assert torch.allclose(second_large, second_small * 4)


@pytest.mark.parametrize("restart_scale", [0.0, -1.0, float("inf"), float("nan")])
def test_rhc_rejects_invalid_restart_scale(restart_scale):
    parameter = nn.Parameter(torch.zeros(1))

    with pytest.raises(ValueError, match="restart_scale"):
        RHC([parameter], restart_scale=restart_scale)


def test_rhc_checkpoint_continues_across_restart_boundary():
    parameter = nn.Parameter(torch.zeros(2))
    optimizer = RHC(
        [parameter],
        step_size=0.2,
        restarts=2,
        restart_interval=2,
        restart_scale=0.4,
        random_state=23,
    )

    def closure():
        return parameter.square().sum()

    optimizer.step(closure)
    optimizer.step(closure)

    serialized = io.BytesIO()
    torch.save(
        {"parameter": parameter, "optimizer": optimizer.state_dict()},
        serialized,
    )
    serialized.seek(0)
    checkpoint = torch.load(serialized, weights_only=True)

    resumed_parameter = nn.Parameter(checkpoint["parameter"].detach().clone())
    resumed = RHC([resumed_parameter], restart_scale=1.0, random_state=999)
    resumed.load_state_dict(checkpoint["optimizer"])

    for _ in range(5):
        loss = optimizer.step(closure)
        resumed_loss = resumed.step(lambda: resumed_parameter.square().sum())
        assert torch.equal(loss, resumed_loss)
        assert torch.equal(parameter, resumed_parameter)

    assert optimizer.completed_restarts == resumed.completed_restarts == 2
    assert optimizer.proposed_steps == resumed.proposed_steps == 4
    assert optimizer.accepted_steps == resumed.accepted_steps
    assert optimizer.rejected_steps == resumed.rejected_steps
    assert resumed.param_groups[0]["restart_scale"] == 0.4
    optimizer.restore_best()
    resumed.restore_best()
    assert torch.equal(parameter, resumed_parameter)
