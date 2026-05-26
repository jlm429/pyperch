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
- `restart_interval`: number of proposed steps between restarts. Use `None` to disable.
- `random_state`: optional seed for reproducible random proposals.

RHC accepts candidate moves only when they do not increase the loss.

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
- `random_state`: optional seed for reproducible random proposals.

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
- `random_state`: optional seed for reproducible population sampling.

GA evolves a population using selection, crossover, and mutation.

---

# Optimizer Counters and State

PyPerch optimizers expose a small set of counters and state values to help inspect optimizer behavior.

- `function_evals`: number of objective/loss evaluations.
- `proposed_steps`: number of candidate updates proposed.
- `accepted_steps`: number of proposed updates accepted.
- `rejected_steps`: number of proposed updates rejected.
- `best_loss`: best loss observed by the optimizer.
- `restore_best()`: restores the best parameter values observed so far.

RHC also exposes:

- `completed_restarts`: number of restarts actually performed.

---

[RHC Examples](../examples/standalone/rhc) 

[SA Examples](../examples/standalone/sa) 

[GA Examples](../examples/standalone/ga) 