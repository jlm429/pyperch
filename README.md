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

## Agent-assisted Experiments

PyPerch's agentic harness supports agent-assisted experiments. One way to try this is with an [orchestrator-worker](https://platform.claude.com/cookbook/patterns-agents-orchestrator-workers) architecture like [FirstMate](https://github.com/kunchenguid/firstmate), which supports multiple coding-agent harnesses, including [Pi](https://pi.dev/).

For example, ask the orchestrator to make a small optimizer change, validate it, and then run parallel experiments:

> Follow `AGENTS.md` and load the relevant skill(s). For local experimentation only, add an optional GA uniform-crossover parameter while preserving the existing `0.5` default, backward compatibility, and native PyTorch usage. Once tests pass, freeze the implementation. Using the Iris dataset and a small PyTorch MLP, run two independent Optuna experiments with the same search space and different random seeds. Report the best trial, best parameters, objective value, number of trials, and limitations for each run, then compare the results. Do not claim the change improves GA unless the evidence supports it. Do not push or open a PR.

---

## Contributing

Pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the project
philosophy, development checks, and review expectations.
