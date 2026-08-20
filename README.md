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

# Examples

The fastest way to get started with PyPerch is to explore the examples.

See:

[Examples](/examples/standalone/)

[Optuna Search](/examples/search/optuna_search_example.py)

---

# Documentation

See:

[General Usage Guide](docs/general_usage_guide.md)

[Search Usage Guide](docs/search.md)

---

# Contributing

Pull requests are welcome.

## Agent-assisted contributions

Compatible coding agents should start with the canonical [agent guide](AGENTS.md).
Claude Code loads the same guide through [`CLAUDE.md`](CLAUDE.md). Task-specific
instructions are disclosed only when needed, such as the
[`add-or-modify-optimizer`](.agents/skills/add-or-modify-optimizer/SKILL.md) skill.

Repository-specific prompts can be concise:

> Follow `AGENTS.md`. Compare `docs/search.md` with `pyproject.toml` and make only the
> documentation corrections needed for accurate Optuna installation guidance.

For a concrete follow-up that exercises the optimizer workflow:

> Follow `AGENTS.md` and the `add-or-modify-optimizer` skill. Extend `GA` with an
> optional uniform-crossover probability for choosing values from the first parent,
> preserving the current hard-coded `0.5` behavior as the default, backward
> compatibility, and native `model.parameters()` usage. Add focused tests, update a
> runnable GA example, and document the option in the general usage guide. Do not
> add framework abstractions or change the Optuna layer unless the concern genuinely
> belongs there.
