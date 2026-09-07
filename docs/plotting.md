# Plot recorded results

`pyperch.plotting` prepares existing numeric results and draws on explicit
caller-owned Matplotlib Axes. It is independent of notebooks, estimators, training
loops, optimizers, and Optuna. It never fits models or computes scores.

Install `pip install 'pyperch[plotting]'` for rendering. NumPy, already a core
requirement, is sufficient for preparation. Importing `pyperch`, its optimizers,
or `pyperch.plotting` does not import Matplotlib, scikit-learn, or Optuna.
Matplotlib is imported only when a renderer is called. Notebook dependencies are
in a separate `notebooks` extra; see the [examples guide](../examples/README.md).

## Choose the question first

| Curve | x | y | Question |
| --- | --- | --- | --- |
| Learning | Training-set sample count | Train and validation performance | Would more representative training data help? |
| Training | Training iteration or measured objective calls | Loss, fitness, or score | Is optimization progressing, stalling, or unstable? |
| Validation | One hyperparameter | Train and validation performance | Which setting generalizes under this protocol? |

Loss versus epochs is a **training curve**, even if another library calls it a
learning curve. A temperature schedule over iterations is a training diagnostic;
a temperature validation curve requires separate trained models at different
initial temperatures. Search history is a different question: use the native
[Optuna diagnostics](https://optuna.readthedocs.io/en/stable/reference/visualization/index.html).
The [tuning notebook](../examples/notebooks/04_optuna_tuning.ipynb) demonstrates it.

## Public signatures

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

plot_learning_curve(data, *, ax, legend=True, line_kwargs=None) -> Axes
plot_training_curve(data, *, ax, legend=True, line_kwargs=None) -> Axes
plot_validation_curve(data, *, ax, legend=True, line_kwargs=None) -> Axes
```

All renderers require `ax` and return **that identical object**. Each accepts only
its matching prepared curve kind. No default Axes is obtained from pyplot.
The public namespace is `pyperch.plotting`; there are no additional top-level
exports.

### Inputs and shapes

- Coordinates are nonempty one-dimensional lists, tuples, or NumPy arrays.
  Numeric coordinates must be finite, strictly increasing, and unique. Nothing
  is sorted silently. A single point is allowed and receives a visible marker.
- `train_sizes` contains positive integer sample counts, not fractions or epochs.
- `iterations` contains nonnegative numeric positions. Use real iteration counts
  or actually measured objective counts and name their units with `x_label`.
  No counts are inferred from an optimizer name or number of calls.
- `param_values` contains ordered numbers or unique nonempty string categories
  in caller order. Numeric values use their true spacing; categories use
  Matplotlib's categorical positions. A connecting line between categories
  indicates order, not a continuous parameter space. Set a numeric log scale on
  your Axes yourself when appropriate.
- Every score array is real, finite, and either `(n_points,)` for one run or
  **`(n_points, n_runs)`**. Rows correspond to x; columns correspond to repeats.
  Boolean, complex, string, empty, ragged, NaN, and infinite score inputs are
  rejected. Explicitly detach tensors and move them to CPU before NumPy conversion.
- Learning and validation curves require both train and validation arrays with
  equal repeat counts. Column j must describe the same fitted model/protocol in
  both. These series are named `Train` and `Validation`.
- Training `scores` is a nonempty mapping such as `{"RHC": rhc_losses,
  "Adam": adam_losses}`. Names must be nonempty strings and cannot begin with `_`,
  which Matplotlib hides in legends. Insertion order controls drawing order.
  Different named series may have different repeat counts, labeled separately.
- All repeats of one series share the supplied x grid. There is no interpolation,
  padding, truncation, missing-value deletion, or smoothing. For different
  optimizer grids, prepare each separately and overlay on the same Axes. For
  unequal-length runs, choose a scientifically defensible common grid yourself
  or display each run separately; do not pad with invented measurements.

`metric` is a required nonempty label such as `"Cross-entropy (nats)"`,
`"Accuracy (fraction)"`, or `"MSE (target units squared)"`. It does not select or
calculate a metric. Values are not negated, normalized, or clipped. State which
direction is better. Negative R² or fitness values are valid. If you have
negative MSE from a scoring library, explicitly convert it before preparing a
positive-MSE plot. Each Axes should compare commensurate metrics.

### Repeats and uncertainty

The line is the arithmetic mean of repeat columns at each x, with equal weight
per column. Repeat counts are not sample counts and are not inferred from seeds.

| `uncertainty` | Bounds with at least two repeats | Legend |
| --- | --- | --- |
| `"std"` (default) | Mean ± sample SD, `ddof=1` | `mean +/- SD, n=...` |
| `"range"` | Observed minimum to maximum | `mean; min/max, n=...` |
| `None` | No bounds | `mean, n=...` |

One repeat always gives a `single run, n=1` line and **no band**, regardless of
mode. Bands describe the supplied runs; they are not confidence intervals,
standard errors, predictions, or significance tests. SD bounds can extend outside
valid metric ranges, including below zero loss or above one accuracy. They are
not clipped because that would change their meaning. Use ranges if observed
extrema answer the question better. Extreme values whose summaries overflow are
rejected; rescale units explicitly.

Document seeds, initialization, split, preprocessing, model, budget, checkpoint
selection, and what varies across repeats. A seed is experimental metadata,
not proof of independence. Overlapping folds, reused validation sets, and paired
initializations limit statistical interpretations. Do not average fold means
as if they had sample weights when your target is a pooled sample metric.

### Prepared values and return structure

Preparation returns a frozen `CurveData` dataclass containing immutable tuples:

| Field | Meaning |
| --- | --- |
| `kind` | `"learning"`, `"training"`, or `"validation"` |
| `x` | Tuple of numeric positions or string categories, length `n_points` |
| `series` | Tuple of `CurveSeries` in drawing order |
| `x_label`, `metric` | Axis labels |
| `uncertainty` | Requested mode, retained even for single runs |

Each `CurveSeries` has `name`, `mean`, `lower`, `upper`, and `n_runs`. Summary tuples
have length `n_points`; absent bounds are `None`. Input arrays are not modified or
retained. Keep raw results and protocol metadata yourself if you need them later.
Use preparation functions to construct these data objects, rather than calling
the dataclass constructors or replacing their fields manually. Malformed inputs
raise `ValueError`; absent or invalid Axes raises `TypeError`.

### Axes composition and ownership

A renderer adds lines, matching translucent bands, x/y labels, and optionally an
Axes legend. Existing artists and titles remain. `legend=False` leaves the
existing legend untouched, allowing the caller to build a combined legend.
`line_kwargs` maps series names to Matplotlib `Axes.plot` style dictionaries, such
as `{"Train": {"color": "navy", "linestyle": ":"}}`. Unspecified names keep default
styles. Unknown series names and `label`, `data`, or `transform` overrides are
rejected because they could hide the summary meaning or change coordinates.
Bands follow the rendered line color. You can further customize returned Axes
artists through native Matplotlib.

Renderers never create figures, call show/save/close, resize, apply figure layout,
clear Axes, change titles/scales, or force limits. Normal Matplotlib autoscaling
may expand unset limits when artists are added. Explicit limits remain under the
caller's control. Shared-axis behavior is ordinary Matplotlib behavior.

```python
import matplotlib.pyplot as plt
from pyperch.plotting import prepare_learning_curve, plot_learning_curve

# Recorded results, rows = sample counts, columns = repeated fits.
curve = prepare_learning_curve(
    [50, 100, 200],
    [[0.95, 0.93], [0.91, 0.92], [0.90, 0.91]],
    [[0.72, 0.76], [0.81, 0.79], [0.86, 0.85]],
    metric="Accuracy (fraction)",
)
fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
assert plot_learning_curve(curve, ax=left) is left
left.set_title("Recorded model performance")
right.text(0.5, 0.5, "Caller-owned panel", ha="center")
# The application chooses whether to display, export, or close this figure:
# fig.savefig("learning.pdf")
```

This small structure example uses illustrative values. For actual executed data,
see [learning and validation](../examples/notebooks/03_learning_and_validation.ipynb)
and [optimizer comparison](../examples/notebooks/02_optimizer_comparison.ipynb).

## Conceptual and technical background

[Dataquest: Learning Curves for Machine Learning](https://www.dataquest.io/blog/learning-curves-machine-learning/)
was reviewed for the sample-count definition, train/validation gap, and data-size
interpretation. The notebooks treat such patterns as clues: finite optimization
budgets, noise, and distribution shift can complicate bias/variance diagnoses.

[Yellowbrick: Validation Curve](https://www.scikit-yb.org/en/latest/api/model_selection/validation_curve.html)
was reviewed for one-parameter comparisons, score arrays with repeat columns, and
variability bands. PyPerch does not adopt its estimator-fitting visualizer model.

[Matplotlib explicit interfaces](https://matplotlib.org/stable/users/explain/figure/api_interfaces.html)
and [Axes.fill_between](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.fill_between.html)
inform caller-owned composition. [NumPy std](https://numpy.org/doc/stable/reference/generated/numpy.std.html)
defines the explicitly selected `ddof=1` calculation. No Yellowbrick or Dataquest
code is vendored, and neither is a dependency.
