# Learn PyPerch through measured results

Start with the concept, run the experiment cells, then interpret recorded results
with the public [plotting API](../docs/plotting.md). All notebooks run independently
and include actual saved outputs. No dataset downloads are required.

| Order | Notebook | Main question |
| --- | --- | --- |
| 1 | [Native training](notebooks/01_native_training.ipynb) | How do closures, freezing, checkpoints, and hybrid updates fit ordinary PyTorch? |
| 2 | [Optimizer comparison](notebooks/02_optimizer_comparison.ipynb) | Is optimization converging, and what did each run measure and cost? |
| 3 | [Learning and validation](notebooks/03_learning_and_validation.ipynb) | Would more data or a different model capacity help? |
| 4 | [Optuna tuning](notebooks/04_optuna_tuning.ipynb) | How do selection and a final held-out evaluation differ? |

Diagnostics are integrated into the comparison and tuning notebooks. A separate
diagnostics notebook would repeat their setup and detach the plots from their
questions. We retain per-run scores and native Optuna trial history instead of
adding a density plot for three seeds or duplicating Optuna parameter importance.

## Install and execute

Python 3.10 through 3.13 is supported. From a source checkout:

```bash
poetry install --extras notebooks
poetry run jupyter notebook examples/notebooks
```

Or use a virtual environment with a supported Python, for example Python 3.12:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[notebooks]'
.venv/bin/python -m notebook examples/notebooks
```

The `notebooks` extra includes Matplotlib, scikit-learn, Optuna, the notebook UI,
nbconvert, and the Python kernel. The smaller `plotting` extra only adds
Matplotlib. `pip install pyperch` keeps both optional.

To execute **all** notebooks from clean kernels, using the current Poetry Python
and saving outputs in place:

```bash
poetry run python scripts/execute_notebooks.py
```

The runner clears prior outputs, uses a fresh kernel for each notebook, rejects
cell errors, and saves a notebook only after successful execution. It prints
per-notebook time, executed cell count, and figure-output count. An optional
`--report <path.json>` writes those numbers as JSON. No persistent kernel
registration is needed. A standard nbconvert workflow also works when the selected
`python3` kernel points at the environment containing the installed package:

```bash
poetry run jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=180 examples/notebooks/*.ipynb
poetry run jupyter nbconvert --to html --template classic --output-dir build/notebooks \
  examples/notebooks/*.ipynb
```

Budget a few minutes for the full set on a laptop CPU, plus dependency installation
and first kernel/font-cache startup. The 180-second limit applies to each code
cell. Exact measured times and versions are in the
[implementation evidence](../docs/plotting_milestone.md); hardware and dependency
versions can change timings and floating-point results. No GPU is needed or tested.

## Reproducibility and interpretation

Each notebook states its fixed data split, preprocessing, architecture, loss,
training budget, checkpoint policy, seeds, and experimental question before the
experiment. Each prints dependency versions and optimizer file SHA256 prefixes.
Runs use CPU float32, one Torch thread, and deterministic Torch operations.
Optimizer implementations remain unchanged from the remote-default baseline.

The comparison uses three seeds on one fixed split; its SD ribbons describe
initialization and optimizer randomness. The learning/validation notebook varies
subset order and initialization while keeping validation data fixed. Those ribbons
combine both sources. They are descriptive sample SD, not confidence intervals.
The acceptance diagnostic uses the observed range instead. Single-run workflows
have no uncertainty bands. Notebooks keep train/validation paired by seed.

A PyPerch optimizer call is not a gradient update or one population evaluation.
The comparison measures objective calls directly and reports monitoring forwards
separately. Adam's backward work is not included in the evaluation count. Do not
read call-index plots as speed comparisons. Search uses an eight-trial exploratory
budget, preserves every native trial state, and scores test data only after
selection and an independent-seed refit.

Notebook cells own figure size, layout, scales, display, and export. Curves and
bands are Matplotlib artists suitable for PDF/SVG export with `fig.savefig(...)`.
Saved PNG notebook outputs support convenient review. Plot cells carry
descriptive alt text; the documented classic HTML template preserves it. The public PyPerch renderers
always consume existing results and an explicit Axes.

## Complete migration map

All ten `.py` scripts present under `examples/` at baseline `731af77` were reviewed.
Their repeated setup and ad hoc plots are consolidated into four notebooks. The
old scripts are removed; historical versions remain in Git.

| Old path under `examples/` | Preserved workflow and destination |
| --- | --- |
| `standalone/rhc/rhc_classification.py` | RHC classification closure, train/validation loss and accuracy, best restore and counts in notebooks 1 and 2. Scheduled restarts are documented in the optimizer guide; the fixed comparison omits them to avoid a second budget factor. |
| `standalone/rhc/rhc_regression.py` | Simple RHC regression in notebook 1; repeated regression comparison with MSE, R², best restore and counts in notebook 2. |
| `standalone/rhc/rhc_freeze.py` | Frozen native parameters and exact unchanged-weight assertions in notebook 1. Separate loss/accuracy boilerplate is consolidated with transfer and hybrid phases. |
| `standalone/rhc/rhc_hybrid_adam.py` | Frozen first layer, Adam middle layer, RHC head and measured hybrid loss in notebook 1. Fresh RHC per changing conditional objective avoids stale cached losses without relying on reset semantics. |
| `standalone/rhc/transfer_nn_example.py` | Bundled digits, training-only scaling, Adam pretraining, temporary state-dict save/reload, frozen features, RHC head refinement, accuracy plots and assertions in notebook 1. Oversized architecture and 1,000-call run reduced for teaching; no claim of cross-domain transfer. |
| `standalone/sa/sa_classification.py` | Repeated classification, cross-entropy, accuracy/F1, current versus best loss, restore, measured calls, temperature and cumulative acceptance in notebook 2. |
| `standalone/sa/sa_regression.py` | Repeated regression MSE/R² and restoration in notebook 2. Shared SA diagnostics replace the redundant temperature/loss twin-axis chart with separate labeled panels. |
| `standalone/ga/ga_classification.py` | GA classification and current/restored scores in notebook 2; measured evaluation axis makes population cost visible. Population 250 replaced by 12 for practical execution. |
| `standalone/ga/ga_regression.py` | Repeated GA regression and restored R² in notebook 2, reusing the same visible task-specific loop. |
| `search/optuna_search_example.py` | Native OptunaSearch, SA search space, per-trial metadata, best-parameter refit and training curve in notebook 4; now with repeated objectives and an untouched test evaluation. |

No useful workflow is intentionally discarded. Repeated plotting and explanatory
comments move into shared API calls and Markdown. These notebooks do not carry
forward the abandoned GA crossover experiment and do not decide RHC
reset-counter semantics.
