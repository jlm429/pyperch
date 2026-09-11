# General Usage Guide

PyPerch optimizers can be used directly with standard PyTorch models and training loops. The standalone optimizers are meant to behave like drop-in-ish PyTorch optimizers, not as a separate neural-network framework.

You bring your own `torch.nn.Module`, loss function, data tensors, evaluation code, and training loop. PyPerch provides randomized optimization algorithms (RHC, SA, GA) that operate directly on model parameters.

If your model works in PyTorch, it should generally work with PyPerch optimizers.

---

# Minimal Example

```python
import torch
from torch import nn

from pyperch.optim import RHC

model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 2),
)

loss_fn = nn.CrossEntropyLoss()

optimizer = RHC(
    model.parameters(),
    step_size=0.1,
    random_state=42,
)

X = torch.randn(100, 10)
y = torch.randint(0, 2, (100,))


def closure():
    output = model(X)
    return loss_fn(output, y)


for step in range(100):
    optimizer.step(closure)

print("Best loss:", optimizer.best_loss)
```

---

# Closures

PyPerch optimizers use closures similar to second-order PyTorch optimizers.

A closure:

- performs a forward pass
- computes the loss
- returns the loss tensor

Example:

```python
def closure():
    output = model(X_train)
    return loss_fn(output, y_train)
```

The optimizer handles parameter perturbations internally.

---

# Available Optimizers

## Random Hill Climbing (RHC)

```python
from pyperch.optim import RHC

optimizer = RHC(
    model.parameters(),
    step_size=0.1,
    restarts=0,
    restart_interval=None,
    restart_scale=1.0,
    random_state=42,
)
```

Parameters:

- `step_size`: scale of the random parameter perturbation.
- `restarts`: maximum number of random restarts.
- `restart_interval`: number of proposed steps between restarts, regardless of whether
  the proposal at the interval boundary is accepted or rejected. Use `None` to
  disable.
- `restart_scale`: finite, positive standard deviation of the zero-centered Gaussian
  parameter reset. It defaults to `1.0` and can be overridden per parameter group.
- `random_state`: seed for RHC's private random stream. An integer makes proposal
  sampling reproducible without reading or changing PyTorch's global random state.
  `None` creates a freshly seeded private stream.

RHC accepts candidate moves only when they do not increase the loss.
After a restart, the next call evaluates the randomized parameters without counting
another proposed step. Restarts occur at exact multiples of `restart_interval` until
the `restarts` budget is exhausted. They do not discard the global best checkpoint.

---

## Simulated Annealing (SA)

```python
from pyperch.optim import SA

optimizer = SA(
    model.parameters(),
    step_size=0.1,
    temperature=1.0,
    min_temperature=0.1,
    cooling=0.95,
    random_state=42,
)
```

Parameters:

- `step_size`: scale of the random parameter perturbation.
- `temperature`: finite, positive initial annealing temperature.
- `min_temperature`: finite, positive lower bound no greater than `temperature`.
- `cooling`: finite multiplicative cooling rate in the interval `(0, 1]`, applied
  after each proposal.
- `random_state`: seed for SA's private random stream. An integer makes proposal and
  acceptance sampling reproducible without reading or changing PyTorch's global
  random state. `None` creates a freshly seeded private stream.

SA may temporarily accept worse solutions while the temperature is high.

---

## Genetic Algorithm (GA)

```python
from pyperch.optim import GA

optimizer = GA(
    model.parameters(),
    population_size=50,
    mutation_rate=0.1,
    initialization_step_size=0.1,
    mutation_step_size=0.1,
    random_state=42,
)
```

Parameters:

- `population_size`: number of candidate solutions per generation.
- `mutation_rate`: probability that each parameter value is mutated.
- `initialization_step_size`: scale of the Gaussian noise used once to initialize
  the population around the caller's model parameters.
- `mutation_step_size`: scale of Gaussian mutation noise applied to children.
- `random_state`: seed for GA's private random stream. An integer makes population
  sampling reproducible without reading or changing PyTorch's global random state.
  `None` creates a freshly seeded private stream.

The first `step()` evaluates and retains the complete initial population. Later
calls each evolve one generation from the retained population. The best half of the
population, with a minimum of one elite, survives unchanged with its known loss.
Uniform crossover and mutation create the remaining children, and only those
fitness-unknown children call the closure. Stable loss and prior-position ordering
makes tied rankings deterministic. The model parameters follow the best current
individual, while `restore_best()` restores the global best observed across all
generations.

---

# Parameter Groups and Joint Search Settings

PyPerch accepts standard PyTorch parameter-group dictionaries. Group options apply
to the tensors in that group, while settings that define the joint randomized
search remain optimizer-level constructor arguments.

| Optimizer | Per-parameter-group options | Optimizer-level options |
| --- | --- | --- |
| RHC | `step_size`, `restart_scale` | `restarts`, `restart_interval`, `random_state` |
| SA | `step_size` | `temperature`, `min_temperature`, `cooling`, `random_state` |
| GA | `initialization_step_size`, `mutation_step_size`, `mutation_rate` | `population_size`, `random_state` |

For example, two RHC groups can use different proposal scales:

```python
optimizer = RHC(
    [
        {"params": model.features.parameters(), "step_size": 0.02},
        {
            "params": model.classifier.parameters(),
            "step_size": 0.1,
            "restart_scale": 0.5,
        },
    ],
    restarts=2,
    restart_interval=50,
    random_state=42,
)
```

RHC proposes one joint-model move and applies each group's `step_size`. A scheduled
restart applies each group's `restart_scale`; its schedule and budget still describe
the whole model. SA selects one trainable parameter tensor from the joint model,
applies that tensor's group `step_size`, and uses one shared temperature schedule. A
GA individual spans all trainable parameters; initialization and mutation use each
tensor's group `initialization_step_size`, `mutation_step_size`, and `mutation_rate`,
while population size, selection, and crossover operate on the joint population.

Putting an optimizer-level option such as `population_size` or `temperature` in a
parameter-group dictionary raises `ValueError` instead of accepting an ineffective
setting. Effective group values are also validated when groups are constructed or
added. Calling `add_param_group()` after a run has started preserves model values
and the private random stream, but clears counters, cached losses, the best-model
checkpoint, and algorithm lifecycle progress because the joint search space has
changed. The next `step()` initializes a fresh run over all current groups.

---

# Optimizer Counters and State

PyPerch optimizers expose a small set of counters and state values to help inspect optimizer behavior.

- `function_evals`: number of objective/loss evaluations.
- `proposed_steps`: number of optimizer proposals attempted.
- `accepted_steps`: number of proposals that improved or were accepted by the
  algorithm.
- `rejected_steps`: number of proposals that did not improve or were rejected by the
  algorithm.
- `best_loss`: best loss observed by the optimizer.
- `restore_best()`: restores the best parameter values observed so far and allows
  optimization to continue from that restored state.
- `reset_counters()`: clears counters and starts fresh run bookkeeping from the
  current model parameters without changing those parameters or rewinding the private
  random stream. RHC restart progress, SA temperature, and the GA population also
  return to their initial run values.

For RHC and SA, proposals are individual candidate moves. For GA, all three step
counters use generation units after population initialization: `proposed_steps` is
the number of generations evolved, `accepted_steps` is the number whose best member
strictly improved the incumbent, and `rejected_steps` is the number that did not.
Therefore `accepted_steps + rejected_steps == proposed_steps` for every optimizer.
`function_evals` always counts actual closure calls. For an even population of size
`P`, GA initialization uses `P` evaluations and each later generation reuses `P / 2`
elite losses while evaluating `P / 2` new children.

RHC also exposes:

- `completed_restarts`: number of restarts actually performed.

## Saving and resuming

Save both the model and optimizer state, just as with a standard PyTorch optimizer:

```python
torch.save(
    {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
    "checkpoint.pt",
)

checkpoint = torch.load("checkpoint.pt", weights_only=True)
model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])
```

Create the receiving optimizer with the same number of parameter groups and the
same number of parameters in each group, in the same order, before loading. Loading
restores per-group settings and the defaults used by future groups, counters,
current and best losses, the best-model parameters, and the private random-generator
position. It also restores RHC restart settings and progress, the SA temperature
schedule and current temperature, or the GA population size, nested individual
parameter tensors, and corresponding loss list. Continued calls therefore follow
the same stochastic trajectory as an uninterrupted compatible run.

Optimizer state dictionaries created by older PyPerch versions did not contain run
or random-generator state. They remain loadable as fresh-run bookkeeping where the
old group structure is compatible, but the unavailable stochastic trajectory cannot
be reconstructed.

---
# Examples

The executed notebooks keep models, losses, loops, and evaluation visible. Follow
the [notebook environment guide](../examples/README.md) for optional dependencies
and fresh execution commands.

[Native training notebook](../examples/notebooks/01_native_training.ipynb)

[RHC, SA, GA, and Adam comparison](../examples/notebooks/02_optimizer_comparison.ipynb)

[Plotting API](plotting.md) and [all notebooks](../examples/README.md)
