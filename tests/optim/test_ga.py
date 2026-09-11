import io

import pytest
import torch
from torch import nn

from pyperch.optim import GA


def available_devices():
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append(torch.device("mps"))
    return devices


def run_ga_trajectory(random_state, unrelated_global_draws=0):
    torch.manual_seed(1234)
    torch.rand(unrelated_global_draws)

    parameter = nn.Parameter(torch.zeros(4))
    optimizer = GA(
        [parameter],
        population_size=6,
        mutation_rate=1.0,
        initialization_step_size=0.25,
        mutation_step_size=0.25,
        random_state=random_state,
    )

    def closure():
        return parameter.sum()

    losses = [optimizer.step(closure).detach().clone() for _ in range(3)]
    return torch.stack(losses), parameter.detach().clone()


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
        initialization_step_size=0.05,
        mutation_step_size=0.05,
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
    assert (
        optimizer.accepted_steps + optimizer.rejected_steps == optimizer.proposed_steps
    )
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
        initialization_step_size=0.05,
        mutation_step_size=0.05,
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
        initialization_step_size=0.05,
        mutation_step_size=0.05,
    )

    def closure():
        optimizer.zero_grad()
        return criterion(model(X), y)

    for _ in range(5):
        optimizer.step(closure)

    frozen_after = [p.detach().clone() for p in model[-1].parameters()]

    for before, after in zip(frozen_before, frozen_after):
        assert torch.equal(before, after)


@pytest.mark.parametrize("device", available_devices(), ids=str)
@pytest.mark.parametrize("candidate_is_accepted", [True, False])
def test_ga_preserves_loss_dtype_and_device_after_initialization(
    device, candidate_is_accepted
):
    dtype = torch.float64 if device.type == "cpu" else torch.float32
    parameter = nn.Parameter(torch.zeros(2, dtype=dtype, device=device))
    optimizer = GA(
        [parameter],
        population_size=4,
        mutation_rate=0.5,
        random_state=42,
    )
    evaluations = 0

    def closure():
        nonlocal evaluations
        evaluations += 1
        loss = parameter.square().sum()
        if evaluations > optimizer.population_size:
            loss = loss - 1 if candidate_is_accepted else loss + 1
        return loss

    initial_loss = optimizer.step(closure)
    later_loss = optimizer.step(closure)

    assert initial_loss.dtype == dtype
    assert initial_loss.device == parameter.device
    assert later_loss.dtype == dtype
    assert later_loss.device == parameter.device
    assert optimizer.accepted_steps == int(candidate_is_accepted)
    assert optimizer.rejected_steps == int(not candidate_is_accepted)

    best_parameters = parameter.detach().clone()
    with torch.no_grad():
        parameter.fill_(10)
    optimizer.restore_best()

    assert parameter.dtype == dtype
    assert parameter.device == initial_loss.device
    assert torch.equal(parameter, best_parameters)


def test_ga_reset_counters_starts_fresh_best_state():
    parameter = nn.Parameter(torch.tensor([0.0]))
    optimizer = GA(
        [parameter],
        population_size=2,
        mutation_rate=0.0,
        initialization_step_size=0.01,
        mutation_step_size=0.01,
        random_state=42,
    )

    def closure():
        return parameter.square().sum()

    optimizer.step(closure)

    with torch.no_grad():
        parameter.fill_(5)
    optimizer.reset_counters()
    fresh_loss = optimizer.step(closure)

    assert fresh_loss.item() == 25
    assert optimizer.best_loss == 25
    assert optimizer.function_evals == 2
    assert optimizer.proposed_steps == 0
    assert optimizer.accepted_steps == 0
    assert optimizer.rejected_steps == 0

    with torch.no_grad():
        parameter.fill_(7)
    optimizer.restore_best()

    assert parameter.item() == 5


def test_ga_same_seed_is_reproducible_and_global_rng_independent():
    losses_a, parameters_a = run_ga_trajectory(42, unrelated_global_draws=1)
    losses_b, parameters_b = run_ga_trajectory(42, unrelated_global_draws=100)

    assert torch.equal(losses_a, losses_b)
    assert torch.equal(parameters_a, parameters_b)


def test_ga_different_seeds_diverge():
    losses_a, parameters_a = run_ga_trajectory(1)
    losses_b, parameters_b = run_ga_trajectory(2)

    assert not torch.equal(losses_a, losses_b)
    assert not torch.equal(parameters_a, parameters_b)


@pytest.mark.parametrize("random_state", [42, None])
def test_ga_does_not_perturb_global_rng(random_state):
    torch.manual_seed(1234)
    state_before = torch.random.get_rng_state()

    run_ga_trajectory(random_state)

    assert torch.equal(torch.random.get_rng_state(), state_before)


def test_ga_none_uses_a_fresh_private_random_stream():
    losses_a, parameters_a = run_ga_trajectory(None)
    losses_b, parameters_b = run_ga_trajectory(None)

    assert not torch.equal(losses_a, losses_b)
    assert not torch.equal(parameters_a, parameters_b)


def checkpoint_population(optimizer):
    state = optimizer.state_dict()["state"]["_pyperch"]["algorithm"]
    return state["population"], state["population_losses"]


def test_ga_persists_elites_and_reuses_their_fitness_for_three_generations():
    parameter = nn.Parameter(torch.tensor([2.0, -1.0]))
    optimizer = GA(
        [parameter],
        population_size=6,
        mutation_rate=1.0,
        initialization_step_size=0.4,
        mutation_step_size=0.2,
        random_state=8,
    )
    evaluations = 0

    def closure():
        nonlocal evaluations
        evaluations += 1
        return parameter.square().sum()

    optimizer.step(closure)
    assert evaluations == 6
    previous_best = optimizer.best_loss

    for generation in range(1, 4):
        population_before, losses_before = checkpoint_population(optimizer)
        elite_index = min(range(len(losses_before)), key=losses_before.__getitem__)
        elite = [value.clone() for value in population_before[elite_index]]
        elite_loss = losses_before[elite_index]

        optimizer.step(closure)
        population_after, losses_after = checkpoint_population(optimizer)

        assert any(
            loss == elite_loss
            and all(torch.equal(left, right) for left, right in zip(individual, elite))
            for individual, loss in zip(population_after, losses_after)
        )
        assert evaluations == 6 + generation * 3
        assert optimizer.function_evals == evaluations
        assert optimizer.proposed_steps == generation
        assert optimizer.accepted_steps + optimizer.rejected_steps == generation
        assert optimizer.best_loss <= previous_best
        previous_best = optimizer.best_loss


def test_ga_checkpoint_round_trip_preserves_population_and_fitness():
    parameter = nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = GA(
        [parameter],
        population_size=6,
        mutation_rate=0.5,
        initialization_step_size=0.4,
        mutation_step_size=0.2,
        random_state=31,
    )
    for _ in range(4):
        optimizer.step(lambda: parameter.square().sum())

    population_before, losses_before = checkpoint_population(optimizer)
    serialized = io.BytesIO()
    torch.save(
        {"parameter": parameter, "optimizer": optimizer.state_dict()},
        serialized,
    )
    serialized.seek(0)
    checkpoint = torch.load(serialized, weights_only=True)

    resumed_parameter = nn.Parameter(checkpoint["parameter"].detach().clone())
    resumed = GA([resumed_parameter], population_size=2, random_state=999)
    resumed.load_state_dict(checkpoint["optimizer"])
    population_after, losses_after = checkpoint_population(resumed)

    assert losses_after == losses_before
    assert len(population_after) == len(population_before) == 6
    for individual_after, individual_before in zip(
        population_after,
        population_before,
    ):
        assert all(
            torch.equal(value_after, value_before)
            for value_after, value_before in zip(
                individual_after,
                individual_before,
            )
        )

    loss = optimizer.step(lambda: parameter.square().sum())
    resumed_loss = resumed.step(lambda: resumed_parameter.square().sum())
    assert torch.equal(resumed_loss, loss)
    assert torch.equal(resumed_parameter, parameter)
    assert resumed.function_evals == optimizer.function_evals
    assert resumed.proposed_steps == optimizer.proposed_steps


def test_ga_initialization_and_mutation_controls_have_independent_magnitudes():
    def initialize(scale):
        parameter = nn.Parameter(torch.zeros(3))
        optimizer = GA(
            [parameter],
            population_size=4,
            mutation_rate=1.0,
            initialization_step_size=scale,
            mutation_step_size=0.1,
            random_state=5,
        )
        optimizer.step(lambda: parameter.square().sum())
        population, _ = checkpoint_population(optimizer)
        return population

    small = initialize(0.2)
    large = initialize(0.6)
    for small_individual, large_individual in zip(small[1:], large[1:]):
        assert torch.allclose(large_individual[0], small_individual[0] * 3)

    def mutate(scale):
        parameter = nn.Parameter(torch.zeros(1))
        optimizer = GA(
            [parameter],
            population_size=2,
            mutation_rate=1.0,
            initialization_step_size=0.5,
            mutation_step_size=scale,
            random_state=17,
        )
        optimizer.step(lambda: parameter.square().sum())
        initial_population, initial_losses = checkpoint_population(optimizer)
        elite_index = min(range(len(initial_losses)), key=initial_losses.__getitem__)
        elite = initial_population[elite_index][0]
        optimizer.step(lambda: parameter.square().sum())
        population, _ = checkpoint_population(optimizer)
        child = next(
            individual[0]
            for individual in population
            if not torch.equal(individual[0], elite)
        )
        return elite, child

    small_parent, small_child = mutate(0.1)
    large_parent, large_child = mutate(0.3)
    assert torch.equal(small_parent, large_parent)
    assert torch.allclose(large_child - large_parent, (small_child - small_parent) * 3)


def test_ga_restore_best_after_multiple_generations():
    parameter = nn.Parameter(torch.tensor([3.0, -2.0]))
    optimizer = GA(
        [parameter],
        population_size=6,
        mutation_rate=1.0,
        initialization_step_size=0.5,
        mutation_step_size=0.25,
        random_state=12,
    )
    for _ in range(5):
        optimizer.step(lambda: parameter.square().sum())

    best_loss = optimizer.best_loss
    with torch.no_grad():
        parameter.fill_(99)
    optimizer.restore_best()

    assert parameter.square().sum().item() == pytest.approx(best_loss)
    assert optimizer._current_loss == best_loss


def test_ga_old_step_size_argument_is_not_accepted():
    parameter = nn.Parameter(torch.zeros(1))

    with pytest.raises(TypeError, match="step_size"):
        GA([parameter], step_size=0.1)
