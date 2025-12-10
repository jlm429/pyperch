# pyperch

![PyPI](https://img.shields.io/pypi/v/pyperch.svg)
![Python Versions](https://img.shields.io/pypi/pyversions/pyperch.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)
![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)
![Linter: Ruff](https://img.shields.io/badge/lint-ruff-blue.svg)
[![CircleCI](https://dl.circleci.com/status-badge/img/circleci/WH9eaoZnQRJ8SGFDrvqQAd/5meq6x5R3uDA3KSuHARdVk/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/circleci/WH9eaoZnQRJ8SGFDrvqQAd/5meq6x5R3uDA3KSuHARdVk/tree/main)


A lightweight, modular library for **neural network weight optimization** using randomized search algorithms built directly on top of **PyTorch**. Pyperch provides flexible alternatives to gradient-based optimization, enabling experimentation with Randomized Hill Climbing, Simulated Annealing, and Genetic Algorithms - as well as hybrid approaches where these methods work alongside traditional optimizers like Adam or SGD.

The current version introduces a unified training API, improved configuration system, deterministic behavior, and a cleaner separation between optimization logic and training.

---

## Key Features

- **Randomized Optimization Algorithms**
  - Randomized Hill Climbing (RHC)
  - Simulated Annealing (SA)
  - Genetic Algorithm (GA)

- **Hybrid Training Support**  
  Combine layer-wise modes (freeze, grad, meta) to mix gradient-free and gradient-based optimization in the same network.

- **Unified Trainer API**  
  An interface for classification, regression, batching, metrics, early stopping, and reproducibility.

- **Pure PyTorch (No Skorch Dependency)**  
  All examples are built on native PyTorch modules and DataLoader.

- **Modern Configuration System**  
  Structured configs (`TrainConfig`, `OptimizerConfig`, etc.) keep experiments consistent and explicit.

- **Utility Functions Included**  
  Metrics, plotting helpers, seed control, and structured outputs.

- **Search Integration**
  Optuna-based hyperparameter grid search (parallel-ready) for RHC/SA/GA tuning.

- **Pure PyTorch**
  No Skorch dependency; all examples use native PyTorch modules and DataLoader.
  
- **Modern Project Tooling**
  - **Poetry** for dependencies, builds, and publishing  
  - **Black** for code formatting  
  - **Ruff** for linting and import sorting  
  - **CircleCI** for automated testing 

- **Utilities Included**
  Metrics, plotting helpers, consistent seed control, and structured training outputs.
  
---

## Installation

```bash
pip install pyperch
```

If developing locally:

```bash
poetry install
```

---

## Quick Start and Examples (Coming Soon)

will be added after the refactor merge:

---

## Legacy Standalone Optimizers (RHC, SA, GA)

If you are upgrading from Pyperch ≤ 0.1.6, the original standalone (functional) optimizers have been preserved for backward compatibility.

You can find the previous implementations here:

- **Git tag:** `<TAG>`  
- **Directory:** `<PATH>` (archived copy from pre-refactor branch)

The new refactored optimizers can be found under:

```
pyperch.optim.*
```

---

## Contributing

Contributions are welcome. To submit a change:

1. Fork the repository  
2. Create a feature branch:

```bash
git checkout -b feature/my-change
```

3. Commit your work:

```bash
git commit -m "feat: describe your change"
```

4. Push your branch:

```bash
git push origin feature/my-change
```

5. Open a pull request on GitHub

---

## Code Style

Before opening a PR:

```bash
poetry run black pyperch
poetry run ruff check pyperch --fix
```

This ensures consistent formatting and linting across the project.

---

## License

MIT License
