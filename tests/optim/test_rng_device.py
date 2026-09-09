import pytest
import torch
from torch import nn

from pyperch.optim import RHC, SA

DEVICE_CASES = [
    pytest.param(torch.device("cpu"), torch.float32, id="cpu-float32"),
    pytest.param(torch.device("cpu"), torch.float64, id="cpu-float64"),
    pytest.param(
        torch.device("cuda"),
        torch.float32,
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA is not available"
        ),
        id="cuda-float32",
    ),
    pytest.param(
        torch.device("mps"),
        torch.float32,
        marks=pytest.mark.skipif(
            not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
            reason="MPS is not available",
        ),
        id="mps-float32",
    ),
]


def run_trajectory(optimizer_type, random_state, unrelated_global_draws=0):
    torch.manual_seed(1234)
    torch.rand(unrelated_global_draws)

    parameter = nn.Parameter(torch.zeros(4))
    optimizer = optimizer_type([parameter], step_size=0.25, random_state=random_state)

    def closure():
        return parameter.sum()

    losses = [optimizer.step(closure).detach().clone() for _ in range(3)]
    return torch.stack(losses), parameter.detach().clone()


@pytest.mark.parametrize("optimizer_type", [RHC, SA])
@pytest.mark.parametrize("device,dtype", DEVICE_CASES)
@pytest.mark.parametrize("candidate_is_accepted", [True, False])
def test_rhc_and_sa_preserve_device_and_dtype_through_randomized_paths(
    optimizer_type, device, dtype, candidate_is_accepted
):
    parameter = nn.Parameter(torch.zeros(2, device=device, dtype=dtype))
    kwargs = {"step_size": 0.1, "random_state": 42}
    if optimizer_type is RHC:
        kwargs.update(restarts=1, restart_interval=1)
    optimizer = optimizer_type([parameter], **kwargs)
    evaluations = 0

    def closure():
        nonlocal evaluations
        evaluations += 1
        loss = parameter.square().sum()
        if candidate_is_accepted:
            return loss + int(evaluations == 1)
        return loss + int(evaluations > 1)

    initial_loss = optimizer.step(closure)
    later_loss = optimizer.step(closure)

    assert initial_loss.dtype == dtype
    assert initial_loss.device == parameter.device
    assert later_loss.dtype == dtype
    assert later_loss.device == parameter.device
    assert optimizer.accepted_steps == int(candidate_is_accepted)
    assert optimizer.rejected_steps == int(not candidate_is_accepted)
    if optimizer_type is RHC:
        assert optimizer.completed_restarts == 1

    optimizer.restore_best()
    assert parameter.dtype == dtype
    assert parameter.device.type == device.type
    assert parameter.square().sum().item() == pytest.approx(optimizer.best_loss)

    continued_loss = optimizer.step(closure)
    assert continued_loss.dtype == dtype
    assert continued_loss.device == parameter.device


@pytest.mark.parametrize("optimizer_type", [RHC, SA])
def test_rhc_and_sa_same_seed_are_reproducible_and_global_rng_independent(
    optimizer_type,
):
    losses_a, parameters_a = run_trajectory(
        optimizer_type, 42, unrelated_global_draws=1
    )
    losses_b, parameters_b = run_trajectory(
        optimizer_type, 42, unrelated_global_draws=100
    )

    assert torch.equal(losses_a, losses_b)
    assert torch.equal(parameters_a, parameters_b)


@pytest.mark.parametrize("optimizer_type", [RHC, SA])
def test_rhc_and_sa_different_seeds_diverge(optimizer_type):
    losses_a, parameters_a = run_trajectory(optimizer_type, 1)
    losses_b, parameters_b = run_trajectory(optimizer_type, 2)

    assert not torch.equal(losses_a, losses_b)
    assert not torch.equal(parameters_a, parameters_b)


@pytest.mark.parametrize("optimizer_type", [RHC, SA])
@pytest.mark.parametrize("random_state", [42, None])
def test_rhc_and_sa_do_not_perturb_global_rng(optimizer_type, random_state):
    torch.manual_seed(1234)
    state_before = torch.random.get_rng_state()

    run_trajectory(optimizer_type, random_state)

    assert torch.equal(torch.random.get_rng_state(), state_before)


@pytest.mark.parametrize("optimizer_type", [RHC, SA])
def test_rhc_and_sa_none_use_fresh_private_random_streams(optimizer_type):
    def run_accepted_proposal():
        parameter = nn.Parameter(torch.zeros(4))
        optimizer = optimizer_type([parameter], step_size=0.25, random_state=None)

        def closure():
            return torch.zeros(())

        optimizer.step(closure)
        optimizer.step(closure)
        assert optimizer.accepted_steps == 1
        return parameter.detach().clone()

    parameters_a = run_accepted_proposal()
    parameters_b = run_accepted_proposal()

    assert not torch.equal(parameters_a, parameters_b)


@pytest.mark.parametrize("optimizer_type", [RHC, SA])
def test_rhc_and_sa_reset_counters_do_not_rewind_private_rng(optimizer_type):
    reset_parameter = nn.Parameter(torch.zeros(4))
    reference_parameter = nn.Parameter(torch.zeros(4))
    reset_optimizer = optimizer_type([reset_parameter], step_size=0.25, random_state=42)
    reference_optimizer = optimizer_type(
        [reference_parameter], step_size=0.25, random_state=42
    )

    def reset_closure():
        return torch.zeros(())

    def reference_closure():
        return torch.zeros(())

    for optimizer, closure in (
        (reset_optimizer, reset_closure),
        (reference_optimizer, reference_closure),
    ):
        optimizer.step(closure)
        optimizer.step(closure)

    reset_optimizer.reset_counters()
    reset_optimizer.step(reset_closure)
    reset_optimizer.step(reset_closure)
    reference_optimizer.step(reference_closure)

    assert torch.equal(reset_parameter, reference_parameter)
