# pyperch

![PyPI](https://img.shields.io/pypi/v/pyperch.svg)
![Python Versions](https://img.shields.io/pypi/pyversions/pyperch.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)
![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)
![Linter: Ruff](https://img.shields.io/badge/lint-ruff-blue.svg)
[![CircleCI](https://dl.circleci.com/status-badge/img/circleci/WH9eaoZnQRJ8SGFDrvqQAd/5meq6x5R3uDA3KSuHARdVk/tree/master.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/circleci/WH9eaoZnQRJ8SGFDrvqQAd/5meq6x5R3uDA3KSuHARdVk/tree/master)


A lightweight library for neural network weight optimization using randomized search algorithms with PyTorch.  PyPerch includes optional hyperparameter search utilities layered on top of the standalone optimizers.

## Installation

Install from PyPI:

```bash
pip install pyperch
```

or with Poetry:

```bash
poetry add pyperch
```

---

## Agent-assisted Experiments

PyPerch's agentic harness supports agent-assisted experiments. One way to try this is with an [orchestrator-worker](https://platform.claude.com/cookbook/patterns-agents-orchestrator-workers) architecture like [FirstMate](https://github.com/kunchenguid/firstmate), which supports multiple coding-agent harnesses, including [Pi](https://pi.dev/).

For example, give the orchestrator a task that combines implementation, validation, and parallel experimentation:

> Follow `AGENTS.md` and load the relevant skill(s). For local experimentation only, add an optional GA uniform-crossover parameter controlling the probability of selecting from the first parent, preserving the existing `0.5` default, backward compatibility, and native PyTorch usage. Add or update focused tests and, once they pass, freeze the implementation. Using the Iris dataset and a small PyTorch MLP, spawn two separate experiment agents to independently run the existing Optuna search workflow with the same reasonable search space, including the new crossover parameter, but different random seeds. Experiment agents must not modify the implementation. Report each run's seed, number of trials, best trial, best parameters, objective value, and any limitations, then briefly compare the results. Do not claim the new parameter improves GA unless the evidence supports it. Do not push, commit, open a PR, or modify the remote repository.

---

## Development Setup

Clone the repository:

```bash
git clone https://github.com/jlm429/pyperch.git
cd pyperch
```

Install development dependencies:

```bash
poetry install
```

---

## Examples

The fastest way to get started with PyPerch is to explore the examples.

See:

[Examples](/examples/standalone/)

[Optuna Search](/examples/search/optuna_search_example.py)

---

## Documentation

See:

[General Usage Guide](docs/general_usage_guide.md)

[Search Usage Guide](docs/search.md)

---

## Contributing

Pull requests are welcome.

