# Optuna Search Usage Guide

The `pyperch.search` package provides an optional, lightweight wrapper for Optuna-based hyperparameter search.

This layer is intended for quick experimentation, tuning, demos, and research workflows while remaining fully compatible with standard PyTorch training patterns. 
PyPerch search utilities do not abstract or replace PyTorch training loops.  
Optuna is an optional dependency. Calling `OptunaSearch.search(...)` without Optuna installed raises an `ImportError`.

---

# Installation

Install the optional Optuna dependency:

```bash
poetry install --extras optuna
```

---

# Primary Usage

```python
from pyperch.search import OptunaSearch

search = OptunaSearch(
    param_space={
        "step_size": ("float", 0.001, 0.1, True),
        "temperature": ("float", 0.1, 5.0, True),
        "cooling": ("float", 0.90, 0.999),
    },
    direction="minimize",
)

study = search.search(objective, n_trials=20)
```

---

# Objective Function

The objective function receives:

```python
params
trial
```

Example:

```python
def objective(params, trial):

    model = make_model()

    optimizer = SA(
        model.parameters(),
        step_size=params["step_size"],
        temperature=params["temperature"],
        cooling=params["cooling"],
    )

    for _ in range(2000):

        def closure():
            output = model(X_train)
            return loss_fn(output, y_train)

        optimizer.step(closure)

    with torch.no_grad():
        valid_loss = loss_fn(model(X_valid), y_valid).item()

    return valid_loss
```

---

# Parameter Space Format

Parameter spaces are defined using tuples.

## Float parameters

```python
("float", low, high)
("float", low, high, log_scale)
```

## Integer parameters

```python
("int", low, high)
("int", low, high, step)
```

## Categorical parameters

```python
("categorical", [choices])
```

Example:

```python
param_space = {
    "step_size": ("float", 0.001, 0.1, True),
    "temperature": ("float", 0.1, 5.0, True),
    "cooling": ("float", 0.90, 0.999),
    "iterations": ("int", 500, 5000, 500),
}
```

---

# Frozen Layers and Existing Models

Because PyPerch search utilities operate directly with PyTorch, existing models and frozen layers work naturally.

Example:

```python
for param in model.features.parameters():
    param.requires_grad = False
```

Only trainable parameters need to be passed to the optimizer:

```python
optimizer = SA(
    filter(lambda p: p.requires_grad, model.parameters()),
    ...
)
```

---

# Return Value

`search.search(...)` returns the underlying Optuna `Study`.

Common attributes:

```python
study.best_params
study.best_value
study.best_trial
study.trials
```

---

# Direction

Supported directions:

```python
"minimize"
"maximize"
```

---

# Examples


[optuna_search_example.py](../examples/optuna_search_example.py) 