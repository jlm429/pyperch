# Examples

Four independent notebooks show ordinary PyTorch training with PyPerch optimizers,
plotting utilities, and Optuna. Each includes saved results and uses generated or
bundled data, with no dataset downloads.

| Notebook | Description |
| --- | --- |
| [1. Native training](notebooks/01_native_training.ipynb) | Closures, checkpoints, frozen layers, and combined Adam/RHC updates. |
| [2. Optimizer comparison](notebooks/02_optimizer_comparison.ipynb) | RHC, SA, GA, and Adam convergence, measured objective calls, and variation across seeds. |
| [3. Learning and validation curves](notebooks/03_learning_and_validation.ipynb) | Training sample counts, model capacity, and generalization. |
| [4. Optuna tuning](notebooks/04_optuna_tuning.ipynb) | Repeated trial objectives, search history, and a final test evaluation. |

## Install and run

From a source checkout with Python 3.10 through 3.13:

```bash
poetry install --extras notebooks
poetry run jupyter notebook examples/notebooks
```

Open a notebook and run all cells in order. Allow a few minutes for the complete
set on a laptop CPU, plus installation and first-startup time. No GPU is needed.

## Reproducibility

Each notebook specifies seeds, splits, preprocessing, architecture, metrics, and
training budgets, and prints dependency versions. Runs use float32 on CPU with
one Torch thread and deterministic operations. Results and timings can vary with
hardware and dependency versions.

Three-seed curves show means with sample-standard-deviation bands unless labeled
as an observed range. These describe variation across runs, not confidence
intervals; single runs have no band. Equal optimizer-call counts do not imply
equal computation. The comparison notebook reports measured objective calls
separately from monitoring and explains the excluded backward work.
