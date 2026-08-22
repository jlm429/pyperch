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

The fastest way to get started with PyPerch is to explore the examples.

See:

[Examples](examples/standalone/)

[Optuna Search](examples/search/optuna_search_example.py)

---

## Documentation

See:

[General Usage Guide](docs/general_usage_guide.md)

[Search Usage Guide](docs/search.md)

---

## Contributing

Pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the project
philosophy, development checks, and review expectations.
