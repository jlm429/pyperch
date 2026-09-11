import io

import pytest
import torch
from torch import nn

from pyperch.optim import GA, RHC, SA

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


class ParameterPair(nn.Module):
    def __init__(self, *, device, dtype):
        super().__init__()
        self.first = nn.Parameter(torch.zeros(4, device=device, dtype=dtype))
        self.second = nn.Parameter(torch.zeros(4, device=device, dtype=dtype))

    def loss(self):
        return (self.first - 0.75).square().sum() + (self.second + 0.25).square().sum()


def make_group_optimizer(optimizer_type, first, second, *, resumed=False):
    first_step = 0.8 if resumed else 0.05
    second_step = 0.7 if resumed else 0.3
    groups = [
        {"params": [first], "step_size": first_step},
        {"params": [second], "step_size": second_step},
    ]

    if optimizer_type is RHC:
        return RHC(
            groups,
            step_size=0.9 if resumed else 0.12,
            restarts=0 if resumed else 2,
            restart_interval=None if resumed else 2,
            random_state=999 if resumed else 17,
        )
    if optimizer_type is SA:
        return SA(
            groups,
            step_size=0.9 if resumed else 0.12,
            temperature=9.0 if resumed else 4.0,
            min_temperature=0.9 if resumed else 0.25,
            cooling=1.0 if resumed else 0.5,
            random_state=999 if resumed else 17,
        )

    groups[0]["mutation_rate"] = 0.0 if resumed else 0.2
    groups[1]["mutation_rate"] = 0.0 if resumed else 0.9
    for group in groups:
        group["initialization_step_size"] = group.pop("step_size")
        group["mutation_step_size"] = group["initialization_step_size"]
    return GA(
        groups,
        population_size=4 if resumed else 6,
        mutation_rate=0.0 if resumed else 0.4,
        initialization_step_size=0.9 if resumed else 0.12,
        mutation_step_size=0.8 if resumed else 0.11,
        random_state=999 if resumed else 17,
    )


def run_constant_loss_trajectory(optimizer_type, second_step_size):
    first = nn.Parameter(torch.zeros(4))
    second = nn.Parameter(torch.zeros(4))
    optimizer = optimizer_type(
        [
            {"params": [first], "step_size": 0.1},
            {"params": [second], "step_size": second_step_size},
        ],
        random_state=11,
    )

    step_count = 21 if optimizer_type is SA else 2
    for _ in range(step_count):
        optimizer.step(lambda: torch.zeros(()))

    return first.detach().clone(), second.detach().clone()


@pytest.mark.parametrize("optimizer_type", [RHC, SA])
def test_rhc_and_sa_apply_distinct_step_size_per_parameter_group(optimizer_type):
    uniform_first, uniform_second = run_constant_loss_trajectory(optimizer_type, 0.1)
    grouped_first, grouped_second = run_constant_loss_trajectory(optimizer_type, 0.7)

    assert torch.equal(grouped_first, uniform_first)
    assert torch.allclose(grouped_second, uniform_second * 7)


def run_ga_group_trajectory(*, second_scale, second_mutation_rate, steps=2):
    first = nn.Parameter(torch.zeros(4))
    second = nn.Parameter(torch.zeros(4))
    optimizer = GA(
        [
            {
                "params": [first],
                "initialization_step_size": 0.1,
                "mutation_step_size": 0.1,
                "mutation_rate": 1.0,
            },
            {
                "params": [second],
                "initialization_step_size": second_scale,
                "mutation_step_size": second_scale,
                "mutation_rate": second_mutation_rate,
            },
        ],
        population_size=6,
        random_state=11,
    )

    for _ in range(steps):
        optimizer.step(lambda: first.sum())

    return first.detach().clone(), second.detach().clone()


def test_ga_applies_distinct_initialization_and_mutation_scales_per_group():
    uniform_first, uniform_second = run_ga_group_trajectory(
        second_scale=0.1,
        second_mutation_rate=1.0,
    )
    grouped_first, grouped_second = run_ga_group_trajectory(
        second_scale=0.7,
        second_mutation_rate=1.0,
    )

    assert torch.equal(grouped_first, uniform_first)
    assert torch.allclose(grouped_second, uniform_second * 7)


def test_ga_applies_distinct_mutation_rate_per_parameter_group():
    unmutated_first, unmutated_second = run_ga_group_trajectory(
        second_scale=0.3,
        second_mutation_rate=0.0,
        steps=6,
    )
    mutated_first, mutated_second = run_ga_group_trajectory(
        second_scale=0.3,
        second_mutation_rate=1.0,
        steps=6,
    )

    assert torch.equal(mutated_first, unmutated_first)
    assert not torch.equal(mutated_second, unmutated_second)


@pytest.mark.parametrize(
    "optimizer_type,option,value",
    [
        (RHC, "random_state", 1),
        (RHC, "restarts", 1),
        (RHC, "restart_interval", 1),
        (SA, "random_state", 1),
        (SA, "temperature", 1.0),
        (SA, "min_temperature", 0.1),
        (SA, "cooling", 0.9),
        (GA, "random_state", 1),
        (GA, "population_size", 4),
    ],
)
def test_joint_search_options_are_rejected_in_parameter_groups(
    optimizer_type, option, value
):
    parameter = nn.Parameter(torch.zeros(1))

    with pytest.raises(ValueError, match="configured on the optimizer"):
        optimizer_type([{"params": [parameter], option: value}])


@pytest.mark.parametrize(
    "optimizer_type,option,value",
    [
        (RHC, "mutation_rate", 0.5),
        (SA, "population_size", 4),
        (GA, "cooling", 0.9),
    ],
)
def test_options_from_other_algorithms_are_rejected_in_parameter_groups(
    optimizer_type, option, value
):
    parameter = nn.Parameter(torch.zeros(1))

    with pytest.raises(ValueError, match="not a parameter-group option"):
        optimizer_type([{"params": [parameter], option: value}])


@pytest.mark.parametrize(
    "optimizer_type,group_options",
    [
        (RHC, {"step_size": 0.0}),
        (SA, {"step_size": -0.1}),
        (GA, {"initialization_step_size": 0.0}),
        (GA, {"mutation_step_size": 0.0}),
        (GA, {"mutation_rate": 1.1}),
    ],
)
def test_effective_group_options_are_validated(optimizer_type, group_options):
    parameter = nn.Parameter(torch.zeros(1))

    with pytest.raises(ValueError):
        optimizer_type([{"params": [parameter], **group_options}])


@pytest.mark.parametrize("optimizer_type", [RHC, SA, GA])
def test_adding_a_parameter_group_starts_fresh_joint_run_bookkeeping(
    optimizer_type,
):
    first = nn.Parameter(torch.zeros(2))
    second = nn.Parameter(torch.zeros(2))
    optimizer = make_group_optimizer(optimizer_type, first, second, resumed=False)

    optimizer.step(lambda: first.square().sum() + second.square().sum())
    optimizer.step(lambda: first.square().sum() + second.square().sum())
    added = nn.Parameter(torch.zeros(2))
    joint_option = {
        RHC: {"restarts": 1},
        SA: {"temperature": 1.0},
        GA: {"population_size": 4},
    }[optimizer_type]
    with pytest.raises(ValueError, match="configured on the optimizer"):
        optimizer.add_param_group({"params": [added], **joint_option})

    new_group = {"params": [added], "step_size": 0.6}
    if optimizer_type is GA:
        new_group.pop("step_size")
        new_group["initialization_step_size"] = 0.6
        new_group["mutation_step_size"] = 0.7
        new_group["mutation_rate"] = 0.8
    optimizer.add_param_group(new_group)

    assert optimizer.function_evals == 0
    assert optimizer.proposed_steps == 0
    assert optimizer.accepted_steps == 0
    assert optimizer.rejected_steps == 0
    assert optimizer.best_loss is None
    if optimizer_type is RHC:
        assert optimizer.completed_restarts == 0
    if optimizer_type is SA:
        assert optimizer.temperature == 4.0

    optimizer.step(
        lambda: first.square().sum() + second.square().sum() + added.square().sum()
    )
    expected_evals = optimizer.population_size if optimizer_type is GA else 1
    assert optimizer.function_evals == expected_evals
    assert optimizer.proposed_steps == 0


@pytest.mark.parametrize("optimizer_type", [RHC, SA, GA])
@pytest.mark.parametrize("device,dtype", DEVICE_CASES)
def test_checkpoint_round_trip_continues_identically(
    optimizer_type,
    device,
    dtype,
):
    uninterrupted_model = ParameterPair(device=device, dtype=dtype)
    uninterrupted = make_group_optimizer(
        optimizer_type,
        uninterrupted_model.first,
        uninterrupted_model.second,
    )

    for _ in range(4):
        uninterrupted.step(uninterrupted_model.loss)

    serialized = io.BytesIO()
    torch.save(
        {
            "model": uninterrupted_model.state_dict(),
            "optimizer": uninterrupted.state_dict(),
        },
        serialized,
    )
    serialized.seek(0)
    checkpoint = torch.load(serialized, map_location="cpu", weights_only=True)

    assert "_pyperch" in checkpoint["optimizer"]["state"]
    if optimizer_type is GA:
        assert all(
            "population_size" not in group
            for group in checkpoint["optimizer"]["param_groups"]
        )

    resumed_model = ParameterPair(device=device, dtype=dtype)
    resumed = make_group_optimizer(
        optimizer_type,
        resumed_model.first,
        resumed_model.second,
        resumed=True,
    )
    resumed_model.load_state_dict(checkpoint["model"])
    resumed.load_state_dict(checkpoint["optimizer"])

    uninterrupted.restore_best()
    resumed.restore_best()

    for _ in range(4):
        uninterrupted_loss = uninterrupted.step(uninterrupted_model.loss)
        resumed_loss = resumed.step(resumed_model.loss)
        assert torch.equal(uninterrupted_loss, resumed_loss)
        assert torch.equal(uninterrupted_model.first, resumed_model.first)
        assert torch.equal(uninterrupted_model.second, resumed_model.second)

    assert resumed.function_evals == uninterrupted.function_evals
    assert resumed.proposed_steps == uninterrupted.proposed_steps
    assert resumed.accepted_steps == uninterrupted.accepted_steps
    assert resumed.rejected_steps == uninterrupted.rejected_steps
    assert resumed.best_loss == uninterrupted.best_loss
    if optimizer_type is RHC:
        assert resumed.param_groups[0]["step_size"] == 0.05
        assert resumed.param_groups[1]["step_size"] == 0.3
        assert resumed.restarts == uninterrupted.restarts == 2
        assert resumed.restart_interval == uninterrupted.restart_interval == 2
        assert resumed.completed_restarts == uninterrupted.completed_restarts
    elif optimizer_type is SA:
        assert resumed.param_groups[0]["step_size"] == 0.05
        assert resumed.param_groups[1]["step_size"] == 0.3
        assert resumed._initial_temperature == uninterrupted._initial_temperature == 4.0
        assert resumed.temperature == uninterrupted.temperature
        assert resumed.min_temperature == uninterrupted.min_temperature == 0.25
        assert resumed.cooling == uninterrupted.cooling == 0.5
    else:
        assert resumed.population_size == uninterrupted.population_size == 6
        assert resumed.param_groups[0]["initialization_step_size"] == 0.05
        assert resumed.param_groups[1]["initialization_step_size"] == 0.3
        assert resumed.param_groups[0]["mutation_step_size"] == 0.05
        assert resumed.param_groups[1]["mutation_step_size"] == 0.3
        assert resumed.param_groups[0]["mutation_rate"] == 0.2
        assert resumed.param_groups[1]["mutation_rate"] == 0.9

    uninterrupted.add_param_group(
        {"params": [nn.Parameter(torch.zeros(1, device=device, dtype=dtype))]}
    )
    resumed.add_param_group(
        {"params": [nn.Parameter(torch.zeros(1, device=device, dtype=dtype))]}
    )
    if optimizer_type is GA:
        assert uninterrupted.param_groups[-1]["initialization_step_size"] == 0.12
        assert resumed.param_groups[-1]["initialization_step_size"] == 0.12
        assert uninterrupted.param_groups[-1]["mutation_step_size"] == 0.11
        assert resumed.param_groups[-1]["mutation_step_size"] == 0.11
        assert uninterrupted.param_groups[-1]["mutation_rate"] == 0.4
        assert resumed.param_groups[-1]["mutation_rate"] == 0.4
    else:
        assert uninterrupted.param_groups[-1]["step_size"] == 0.12
        assert resumed.param_groups[-1]["step_size"] == 0.12


def test_ga_loads_legacy_uniform_population_size_as_optimizer_level_state():
    parameter = nn.Parameter(torch.zeros(1))
    legacy = GA([parameter], population_size=7).state_dict()
    legacy["state"].pop("_pyperch")
    legacy["param_groups"][0]["population_size"] = 7

    restored_parameter = nn.Parameter(torch.zeros(1))
    restored = GA([restored_parameter], population_size=2)
    restored.load_state_dict(legacy)

    assert restored.population_size == 7
    assert "population_size" not in restored.param_groups[0]
    assert restored.function_evals == 0
