# pyperch

![PyPI](https://img.shields.io/pypi/v/pyperch.svg)
![Python Versions](https://img.shields.io/pypi/pyversions/pyperch.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)
![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)
![Linter: Ruff](https://img.shields.io/badge/lint-ruff-blue.svg)
[![CircleCI](https://dl.circleci.com/status-badge/img/circleci/WH9eaoZnQRJ8SGFDrvqQAd/5meq6x5R3uDA3KSuHARdVk/tree/master.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/circleci/WH9eaoZnQRJ8SGFDrvqQAd/5meq6x5R3uDA3KSuHARdVk/tree/master)


A lightweight library for neural network weight optimization using randomized search algorithms built directly on top of PyTorch.  Pyperch is a research and teaching-oriented library for training neural networks using randomized optimization methods (RHC, SA, GA), gradient-based methods, and hybrid combinations.

## Getting Started

Clone the repository:

```bash
git clone https://github.com/jlm429/pyperch.git
cd pyperch
```

Install dependencies with Poetry:

```bash
poetry install
```

Run the test suite:

```bash
poetry run pytest
```

Run linting and formatting checks:

```bash
poetry run ruff format .
poetry run ruff check .
```

Build the package:

```bash
poetry build
```

Run an example:

```bash
poetry run python examples/standalone/rhc/transfer_nn_example.py
```