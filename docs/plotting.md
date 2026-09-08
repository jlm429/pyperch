# Plot recorded results

`pyperch.plotting` separates data preparation from Matplotlib rendering. Preparation
summarizes recorded numeric results. Rendering draws on a caller-provided Axes and
returns that same Axes. The caller controls figure creation, layout, display, saving,
scales, and further customization.

Install the rendering dependency with `pip install 'pyperch[plotting]'`.

## Terminology

| Plot | Meaning | Typical question |
| --- | --- | --- |
| Learning curve | Performance versus number of training samples | Would more training data help? |
| Training curve | Performance, loss, or fitness versus iterations or measured objective calls | Is optimization progressing? |
| Validation curve | Performance versus one hyperparameter | Which setting works best under this protocol? |

Loss versus epochs is a training curve, even when another library uses a different
name. Search history is also a different diagnostic; use the native
[Optuna visualizations](https://optuna.readthedocs.io/en/stable/reference/visualization/index.html)
for that.

## Public API

Preparation functions:

```python
prepare_learning_curve(
    train_sizes, train_scores, validation_scores, *, metric, uncertainty="std"
) -> CurveData

prepare_training_curve(
    iterations, scores, *, metric, x_label="Iteration", uncertainty="std"
) -> CurveData

prepare_validation_curve(
    param_values, train_scores, validation_scores, *, param_name, metric,
    uncertainty="std"
) -> CurveData
```

Matching renderers:

```python
plot_learning_curve(data, *, ax, legend=True, line_kwargs=None) -> Axes
plot_training_curve(data, *, ax, legend=True, line_kwargs=None) -> Axes
plot_validation_curve(data, *, ax, legend=True, line_kwargs=None) -> Axes
```

Import these functions and the prepared data types from `pyperch.plotting`.

## Data to pass

Coordinates are one-dimensional sequences:

- `train_sizes` contains increasing positive integer sample counts.
- `iterations` contains increasing nonnegative positions. Pass actual iterations or
  measured objective calls, then identify the units with `x_label`.
- `param_values` contains increasing numbers or ordered, unique string categories for
  one hyperparameter.

Scores can have shape `(n_points,)` for one run or `(n_points, n_runs)` for repeated
runs. Rows align with the coordinate values and columns identify repeats. Learning
and validation curves take separate train and validation arrays with paired repeat
columns. A training curve takes a mapping of series names to arrays, for example
`{"Train": train_loss, "Validation": validation_loss}` or
`{"RHC": rhc_loss, "Adam": adam_loss}`.

All repeats within one series use the same coordinate grid. Preparation does not fit
models, calculate metrics, align grids, interpolate, or smooth values. The required
`metric` string is the y-axis label, such as `"Accuracy (fraction)"` or
`"Cross-entropy (nats)"`; state in nearby prose whether higher or lower is better.

## Repeated runs and uncertainty

Repeated runs are summarized by their arithmetic mean at each coordinate.

| `uncertainty` | Band |
| --- | --- |
| `"std"` | Mean plus or minus sample standard deviation |
| `"range"` | Observed minimum to maximum |
| `None` | No band |

A single run has no band. These bands describe the supplied runs; they are not
confidence intervals. Keep seeds and other protocol details with your experimental
record so readers know what varied across columns.

## Prepared data

Each preparation function returns an immutable `CurveData` value:

| Field | Contents |
| --- | --- |
| `kind` | `"learning"`, `"training"`, or `"validation"` |
| `x` | Prepared numeric positions or category labels |
| `series` | `CurveSeries` values in drawing order |
| `x_label` | Label for the horizontal axis |
| `metric` | Label for the vertical axis |
| `uncertainty` | Requested band mode |

Each `CurveSeries` contains `name`, `mean`, `lower`, `upper`, and `n_runs`. The
summary arrays are immutable tuples. `lower` and `upper` are `None` when no band is
available. Construct these values through the preparation functions rather than
instantiating the dataclasses directly.

## Prepare, render, and customize

This example plots two repeated measurements of train and validation loss. Lower is
better.

```python
import matplotlib.pyplot as plt

from pyperch.plotting import prepare_training_curve, plot_training_curve

curve = prepare_training_curve(
    [0, 10, 20, 30],
    {
        "Train": [
            [1.42, 1.38],
            [0.91, 0.95],
            [0.63, 0.66],
            [0.49, 0.52],
        ],
        "Validation": [
            [1.47, 1.43],
            [1.02, 1.05],
            [0.81, 0.84],
            [0.76, 0.79],
        ],
    },
    metric="Cross-entropy (nats)",
    x_label="Optimizer calls",
    uncertainty="range",
)

fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
returned_ax = plot_training_curve(
    curve,
    ax=ax,
    line_kwargs={"Validation": {"color": "tab:orange", "linestyle": ":"}},
)
assert returned_ax is ax
ax.set_title("Recorded optimization progress")
ax.set_yscale("log")
fig.savefig("training-curve.png", dpi=150)
```

`line_kwargs` maps series names to normal `Axes.plot` styling options. Renderers add
lines, matching translucent bands, axis labels, and an optional legend while keeping
existing artists and titles. Pass `legend=False` when composing a shared legend.

For executed examples, see [native training](../examples/notebooks/01_native_training.ipynb),
[optimizer comparison](../examples/notebooks/02_optimizer_comparison.ipynb), and
[learning and validation curves](../examples/notebooks/03_learning_and_validation.ipynb).

## Conceptual references

[Dataquest: Learning Curves for Machine Learning](https://www.dataquest.io/blog/learning-curves-machine-learning/)
provides background on sample-count learning curves and train/validation gaps.
[Yellowbrick: Validation Curve](https://www.scikit-yb.org/en/latest/api/model_selection/validation_curve.html)
describes one-parameter validation curves and variability across repeated scores.
PyPerch prepares already-recorded arrays rather than fitting estimator objects.

For figure and Axes composition, see
[Matplotlib's explicit interface](https://matplotlib.org/stable/users/explain/figure/api_interfaces.html).
