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
    random_state=42,
)
```

Parameters:

- `step_size`: scale of the random parameter perturbation.
- `restarts`: maximum number of random restarts.
- `restart_interval`: number of proposed steps between restarts, regardless of whether
  the proposal at the interval boundary is accepted or rejected. Use `None` to
  disable.
- `random_state`: seed for RHC's private random stream. An integer makes proposal
  sampling reproducible without reading or changing PyTorch's global random state.
  `None` creates a freshly seeded private stream.

RHC accepts candidate moves only when they do not increase the loss.
After a restart, the next call evaluates the randomized parameters without counting
another proposed step.

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
- `temperature`: initial annealing temperature.
- `min_temperature`: lower bound for the temperature.
- `cooling`: multiplicative cooling rate applied after each step.
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
    step_size=0.1,
    random_state=42,
)
```

Parameters:

- `population_size`: number of candidate solutions per generation.
- `mutation_rate`: probability that each parameter value is mutated.
- `step_size`: scale of random initialization and mutation noise.
- `random_state`: seed for GA's private random stream. An integer makes population
  sampling reproducible without reading or changing PyTorch's global random state.
  `None` creates a freshly seeded private stream.

GA evolves a population using selection, crossover, and mutation.

---

# Parameter Groups and Joint Search Settings

PyPerch accepts standard PyTorch parameter-group dictionaries. Group options apply
to the tensors in that group, while settings that define the joint randomized
search remain optimizer-level constructor arguments.

| Optimizer | Per-parameter-group options | Optimizer-level options |
| --- | --- | --- |
| RHC | `step_size` | `restarts`, `restart_interval`, `random_state` |
| SA | `step_size` | `temperature`, `min_temperature`, `cooling`, `random_state` |
| GA | `step_size`, `mutation_rate` | `population_size`, `random_state` |

For example, two RHC groups can use different proposal scales:

```python
optimizer = RHC(
    [
        {"params": model.features.parameters(), "step_size": 0.02},
        {"params": model.classifier.parameters(), "step_size": 0.1},
    ],
    restarts=2,
    restart_interval=50,
    random_state=42,
)
```

RHC proposes one joint-model move and applies each group's `step_size`. Its restart
schedule and budget describe the whole model. SA selects one trainable parameter
tensor from the joint model, applies that tensor's group `step_size`, and uses one
shared temperature schedule. A GA individual spans all trainable parameters;
initialization and mutation use each tensor's group `step_size` and `mutation_rate`,
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
- `proposed_steps`: number of candidate updates proposed.
- `accepted_steps`: number of proposed updates accepted.
- `rejected_steps`: number of proposed updates rejected.
- `best_loss`: best loss observed by the optimizer.
- `restore_best()`: restores the best parameter values observed so far and allows
  optimization to continue from that restored state.
- `reset_counters()`: clears counters and starts fresh run bookkeeping from the
  current model parameters without changing those parameters or rewinding the private
  random stream. RHC restart progress and SA temperature also return to their initial
  run values.

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

Create the receiving optimizer with the same parameter-group count, order, and
parameter count before loading. Loading restores per-group settings and their
defaults, counters, current and best losses, the best-model parameters, and the
private random-generator position. It also restores RHC restart settings and
progress, the SA temperature schedule and current temperature, or GA population
size. Continued calls therefore follow the same stochastic trajectory as an
uninterrupted compatible run.

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
