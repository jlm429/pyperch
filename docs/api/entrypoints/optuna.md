# Optuna Search API

Pyperch integrates with Optuna using an adapter-based design to keep the search layer independent of training. 

---

## Core Components

The Optuna integration consists of the following public components:

- `TrainConfigBuilder` - applies parameter overrides to a base training configuration
- `SearchStrategy` - defines how hyperparameters are sampled
- `OptunaStrategy` - Optuna-backed implementation of `SearchStrategy`
- `TrainerAdapter` - exposes training as an Optuna-compatible objective
- `OptunaSearch` - thin wrapper around an Optuna study
- `SearchFactory` - convenience helpers for constructing searches

---

## Typical Workflow

1. Create a base training configuration
2. Define a search strategy (search space)
3. Wrap training logic using `TrainerAdapter`
4. Create and run a search via `SearchFactory`

---

## TrainerAdapter

`TrainerAdapter` connects the search layer to user-defined training code.

It combines:

- a `TrainConfigBuilder`
- a `SearchStrategy`
- a user-supplied training function

into a single callable compatible with `optuna.study.optimize`.

The adapter does not depend on any specific trainer or model class.
Users are free to define how training is performed.

---

## Example: SQLite-backed Search

```python
search = SearchFactory.optuna_sqlite(
    adapter=adapter,
    study_name="sa_search_demo",
    storage="sqlite:///sa_demo.db",
)

study = search.run(
    n_trials=150,
    n_jobs=4,
)
```

---

## OptunaSearch

`OptunaSearch` is a small convenience wrapper around an Optuna `Study`.

It delegates trial evaluation to a `TrainerAdapter` and exposes
the underlying study object directly.

---

### `run(...)`

```python
study = search.run(
    n_trials=100,
    n_jobs=4,
    timeout=None,
)
```

Runs hyperparameter optimization and returns the Optuna study.

---

### Properties

```python
search.best_params
search.best_value
search.best_trial
```

These properties are pass-throughs to the underlying study.

---

## Parameter Override Paths

Search strategies return dictionaries mapping parameter paths to values.
Keys may refer to nested configuration attributes using dotted paths.

Example:

```python
{
    "optimizer_config.step_size": 0.05,
    "max_epochs": 100,
}
```

These overrides are applied by `TrainConfigBuilder` when constructing
a configuration for each trial.

---

## Stability Notes

The following components are public and stable:

- `TrainConfigBuilder`
- `SearchStrategy`
- `OptunaStrategy`
- `TrainerAdapter`
- `OptunaSearch`
- `SearchFactory`

## Examples

For a complete, working example of Optuna-based grid search, see the example notebook:

- **[Optuna Search Example Notebook](../../../examples/Optuna_examples.ipynb)**
