# pyperch

![PyPI](https://img.shields.io/pypi/v/pyperch.svg)
![Python Versions](https://img.shields.io/pypi/pyversions/pyperch.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)
![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)
![Linter: Ruff](https://img.shields.io/badge/lint-ruff-blue.svg)
[![CircleCI](https://dl.circleci.com/status-badge/img/circleci/WH9eaoZnQRJ8SGFDrvqQAd/5meq6x5R3uDA3KSuHARdVk/tree/master.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/circleci/WH9eaoZnQRJ8SGFDrvqQAd/5meq6x5R3uDA3KSuHARdVk/tree/master)


PyPerch provides randomized hill climbing (RHC), simulated annealing (SA), and
genetic algorithm (GA) optimizers for ordinary `torch.nn.Module` workflows. Pass
native PyTorch parameter iterables, define the loss in a closure, and keep model
architecture, data loading, forward passes, metrics, and training loops in PyTorch.

Optional plotting utilities prepare recorded results for caller-owned Matplotlib
Axes. A thin Optuna layer supports hyperparameter studies without hiding native
trials or studies. Start with the [General Usage Guide](docs/general_usage_guide.md)
for optimizer semantics and complete examples.

## Installation

PyPerch supports Python 3.10 through 3.13.

Install from PyPI:

```bash
pip install pyperch
```

or with Poetry:

```bash
poetry add pyperch
```

Optuna search is an optional extra. Install it from PyPI with:

```bash
pip install "pyperch[optuna]"
```

or with Poetry:

```bash
poetry add "pyperch[optuna]"
```

---

## Development Setup

See [CONTRIBUTING.md](CONTRIBUTING.md) for source-checkout setup and validation.

---

## Examples

PyPerch's RHC, SA, and GA optimizers operate directly on the parameters of standard
PyTorch models. Bring your own `torch.nn.Module`, loss function, data tensors,
evaluation code, and training loop. If a model works in PyTorch, it should generally
work with PyPerch optimizers.

- [Native PyTorch training](examples/notebooks/01_native_training.ipynb): Use RHC in a
  PyTorch training loop, including frozen layers and separate Adam/RHC updates.
- [Optimizer convergence and cost](examples/notebooks/02_optimizer_comparison.ipynb):
  Compare RHC, SA, GA, and Adam across repeated runs.
- [Training, learning, and validation curves](examples/notebooks/03_learning_and_validation.ipynb):
  Explore optimization progress, training set size, model capacity, and generalization.
- [Optuna tuning](examples/notebooks/04_optuna_tuning.ipynb): Tune SA and GA settings
  with repeated Optuna trials and evaluate the selected refits.

See the [examples guide](examples/README.md) for notebook-running instructions and
reproducibility information.

---

## Documentation

- [General Usage Guide](docs/general_usage_guide.md): RHC, SA, and GA in ordinary
  PyTorch training loops
- [Executed examples](examples/README.md): optimizer, freezing, composition, and
  comparison workflows
- [Plotting API and terminology](docs/plotting.md): prepare and render recorded
  curves
- [Search Usage Guide](docs/search.md): optional Optuna studies

---

## Contributing

Pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the project
philosophy, development checks, and review expectations.
