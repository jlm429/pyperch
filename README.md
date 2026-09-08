# pyperch

![PyPI](https://img.shields.io/pypi/v/pyperch.svg)
![Python Versions](https://img.shields.io/pypi/pyversions/pyperch.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)
![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)
![Linter: Ruff](https://img.shields.io/badge/lint-ruff-blue.svg)
[![CircleCI](https://dl.circleci.com/status-badge/img/circleci/WH9eaoZnQRJ8SGFDrvqQAd/5meq6x5R3uDA3KSuHARdVk/tree/master.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/circleci/WH9eaoZnQRJ8SGFDrvqQAd/5meq6x5R3uDA3KSuHARdVk/tree/master)


A lightweight library for neural network weight optimization using randomized search
algorithms with PyTorch. PyPerch includes optional hyperparameter search utilities
layered on top of the standalone optimizers.

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

Start with the [executed notebook progression](examples/README.md): native
training and freezing, optimizer comparisons, learning and validation curves,
then Optuna tuning. Each notebook explains what is measured and how to interpret
the plots, and runs independently on a laptop CPU without dataset downloads.

For a source checkout, install and launch with:

```bash
poetry install --extras notebooks
poetry run jupyter notebook examples/notebooks
```

For plots in your own application, install `pip install 'pyperch[plotting]'` and
use the [public plotting API](docs/plotting.md). It prepares existing results and
renders onto your Matplotlib Axes. Learning curves use training-set sample count;
training curves use iterations; validation curves use a hyperparameter.

See the [examples guide](examples/README.md) for notebook setup, practical runtimes,
and reproducibility guidance.

---

## Documentation

See:

[General Usage Guide](docs/general_usage_guide.md)

[Search Usage Guide](docs/search.md)

[Plotting API and terminology](docs/plotting.md)

---

## Contributing

Pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the project
philosophy, development checks, and review expectations.
