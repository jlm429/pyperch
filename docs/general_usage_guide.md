# General Usage Guide 

PyPerch optimizers can be used directly with standard PyTorch models and training loops.  The standalone optimizers are meant to behave like drop-in-ish PyTorch optimizers, not as a separate neural-network framework. You bring your own `torch.nn.Module`, loss function, data tensors, evaluation code, and training loop. PyPerch provides randomized optimization algorithms (RHC, SA, GA) that operate directly on model parameters.  If your model works in PyTorch, it should generally work with PyPerch optimizers.

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
    random_state=42,
)
```

Common parameters:

- `step_size`
- `random_state`

---

## Simulated Annealing (SA)

```python
from pyperch.optim import SA

optimizer = SA(
    model.parameters(),
    step_size=0.05,
    temperature=1.0,
    cooling=0.995,
    min_temperature=0.001,
    random_state=42,
)
```

Additional SA parameters:

- `temperature`
- `cooling`
- `min_temperature`

SA may temporarily accept worse solutions early in training to encourage exploration.

---

## Genetic Algorithm (GA)

```python
from pyperch.optim import GA

optimizer = GA(
    model.parameters(),
    population_size=100,
    mutation_rate=0.1,
    step_size=0.05,
    random_state=42,
)
```

Additional GA parameters:

- `population_size`
- `mutation_rate`

GA evolves a population of candidate solutions rather than a single parameter state.

---

# Optimizer Counters

PyPerch optimizers expose a small set of common counters to help inspect optimizer behavior.

- `function_evals`: number of times the closure/loss function has been evaluated
- `proposed_steps`: number of candidate parameter updates proposed
- `accepted_steps`: number of proposed updates accepted
- `rejected_steps`: number of proposed updates rejected
- `best_loss`: best loss value observed by the optimizer

---

# Example Gallery

[RHC Examples](../examples/standalone/rhc) 

[SA Examples](../examples/standalone/sa) 

[GA Examples](../examples/standalone/ga) 