"""Prepare recorded results and render on caller-owned Matplotlib Axes.

No training, estimator, or pyplot dependency is involved in preparation.
See docs/plotting.md for shapes, repeat semantics, and figure ownership.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from matplotlib.axes import Axes

__all__ = [
    "CurveData",
    "CurveSeries",
    "prepare_learning_curve",
    "prepare_training_curve",
    "prepare_validation_curve",
    "plot_learning_curve",
    "plot_training_curve",
    "plot_validation_curve",
]

Uncertainty = Literal["std", "range"] | None
Kind = Literal["learning", "training", "validation"]


@dataclass(frozen=True)
class CurveSeries:
    """One named summary with immutable point values and repeat count.

    ``mean``, ``lower`` and ``upper`` have length n_points. Bounds are None
    for one run or uncertainty=None. ``n_runs`` counts columns, not samples.
    Construct through the preparation functions.
    """

    name: str
    mean: tuple[float, ...]
    lower: tuple[float, ...] | None
    upper: tuple[float, ...] | None
    n_runs: int


@dataclass(frozen=True)
class CurveData:
    """Prepared result independent of Matplotlib and the original input arrays.

    ``x`` retains numeric order or categorical labels. ``series`` retains mapping
    insertion order. ``metric`` is the y label; no sign conversion is performed.
    Construct through the preparation functions.
    """

    kind: Kind
    x: tuple[float, ...] | tuple[str, ...]
    series: tuple[CurveSeries, ...]
    x_label: str
    metric: str
    uncertainty: Uncertainty


def _label(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string.")
    return value


def _numeric(value: ArrayLike, name: str) -> np.ndarray:
    try:
        array = np.asarray(value)
        if array.dtype.kind not in "iuf":
            raise ValueError
        array = array.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain real numeric values.") from exc
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _coordinates(values: ArrayLike, kind: Kind) -> tuple:
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError("x must be a nonempty one-dimensional array.") from exc
    if array.ndim != 1 or array.size == 0:
        raise ValueError("x must be a nonempty one-dimensional array.")
    if kind == "validation" and array.dtype.kind in "UO":
        labels = np.asarray(values, dtype=object).tolist()
        if not all(isinstance(v, str) and v.strip() for v in labels):
            raise ValueError("Categorical values must all be nonempty strings.")
        if len(set(labels)) != len(labels):
            raise ValueError("Categorical values must be unique.")
        return tuple(labels)
    array = _numeric(values, "x")
    if np.any(array[1:] <= array[:-1]):
        raise ValueError("Numeric x must be strictly increasing, without duplicates.")
    if kind == "learning" and (np.any(array <= 0) or np.any(array != np.floor(array))):
        raise ValueError("Training sizes must be positive integer sample counts.")
    if kind == "training" and np.any(array < 0):
        raise ValueError("Training coordinates must be nonnegative.")
    return tuple(array.tolist())


def _prepare(kind, x, scores, metric, x_label, uncertainty):
    if uncertainty not in (None, "std", "range"):
        raise ValueError("uncertainty must be 'std', 'range', or None.")
    metric = _label(metric, "metric")
    x_label = _label(x_label, "x_label")
    x = _coordinates(x, kind)
    if not isinstance(scores, Mapping) or not scores:
        raise ValueError("scores must be a nonempty mapping of names to arrays.")
    summaries = []
    for name, values in scores.items():
        _label(name, "series name")
        if name.startswith("_"):
            raise ValueError("Series names cannot start with '_' (hidden by legends).")
        values = _numeric(values, name)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2 or values.shape[0] != len(x) or values.shape[1] == 0:
            raise ValueError(
                f"{name} must have shape (n_points,) or (n_points, n_runs), "
                "with at least one run."
            )
        n_runs = values.shape[1]
        lower = upper = None
        with np.errstate(over="ignore", invalid="ignore"):
            mean = values.mean(axis=1)
            if n_runs > 1 and uncertainty == "std":
                spread = values.std(axis=1, ddof=1)
                lower, upper = mean - spread, mean + spread
            elif n_runs > 1 and uncertainty == "range":
                lower, upper = values.min(axis=1), values.max(axis=1)
        if not np.isfinite(mean).all() or (
            lower is not None
            and not (np.isfinite(lower).all() and np.isfinite(upper).all())
        ):
            raise ValueError("Summary overflowed; rescale the metric values.")
        summaries.append(
            CurveSeries(
                name,
                tuple(mean.tolist()),
                None if lower is None else tuple(lower.tolist()),
                None if upper is None else tuple(upper.tolist()),
                n_runs,
            )
        )
    if kind != "training" and len({s.n_runs for s in summaries}) != 1:
        raise ValueError("Train and validation must have the same number of runs.")
    return CurveData(kind, x, tuple(summaries), x_label, metric, uncertainty)


def prepare_learning_curve(
    train_sizes: ArrayLike,
    train_scores: ArrayLike,
    validation_scores: ArrayLike,
    *,
    metric: str,
    uncertainty: Uncertainty = "std",
) -> CurveData:
    """Summarize performance versus positive integer training-set sample counts.

    Scores have shape (n_sizes,) or (n_sizes, n_runs), with paired train and
    validation columns. Values must be finite; sizes must be strictly increasing.
    No fitting occurs. See prepare_training_curve for uncertainty definitions.
    """
    return _prepare(
        "learning",
        train_sizes,
        {"Train": train_scores, "Validation": validation_scores},
        metric,
        "Training samples",
        uncertainty,
    )


def prepare_training_curve(
    iterations: ArrayLike,
    scores: Mapping[str, ArrayLike],
    *,
    metric: str,
    x_label: str = "Iteration",
    uncertainty: Uncertainty = "std",
) -> CurveData:
    """Summarize recorded metrics versus iterations or measured objective calls.

    Each named array has shape (n_points,) for one run or (n_points, n_runs).
    All runs in a series must share the explicit, increasing nonnegative x grid.
    Different series can have different repeat counts. No alignment, missing-value
    removal, smoothing, scoring, or training is performed.

    The line is the arithmetic mean. 'std' gives mean +/- sample SD (ddof=1),
    'range' gives observed min/max, and None omits bounds. A single run has no
    bounds for every mode. These bands are descriptive, not confidence intervals.
    ``metric`` and ``x_label`` are explicit labels, not metric computations.
    """
    return _prepare("training", iterations, scores, metric, x_label, uncertainty)


def prepare_validation_curve(
    param_values: ArrayLike,
    train_scores: ArrayLike,
    validation_scores: ArrayLike,
    *,
    param_name: str,
    metric: str,
    uncertainty: Uncertainty = "std",
) -> CurveData:
    """Summarize paired scores versus one hyperparameter, without fitting.

    Values are strictly increasing finite numbers or unique nonempty string
    categories in caller order. Scores have shape (n_values,) or
    (n_values, n_runs). See prepare_training_curve for uncertainty definitions.
    """
    return _prepare(
        "validation",
        param_values,
        {"Train": train_scores, "Validation": validation_scores},
        metric,
        param_name,
        uncertainty,
    )


def _plot(data, kind, ax, legend, line_kwargs):
    try:
        from matplotlib.axes import Axes
    except ImportError as exc:
        raise ImportError(
            "Rendering requires Matplotlib. "
            "Install with: pip install 'pyperch[plotting]'"
        ) from exc
    if not isinstance(ax, Axes):
        raise TypeError("ax must be an explicit Matplotlib Axes.")
    if not isinstance(data, CurveData) or data.kind != kind:
        raise ValueError(f"Expected prepared {kind} curve data.")
    styles = {} if line_kwargs is None else line_kwargs
    if not isinstance(styles, Mapping) or set(styles) - {s.name for s in data.series}:
        raise ValueError("line_kwargs must map existing series names to line styles.")
    for style in styles.values():
        if not isinstance(style, Mapping) or {"label", "data", "transform"} & set(
            style
        ):
            raise ValueError(
                "Line styles must be mappings without label/data/transform."
            )
    for series in data.series:
        description = "single run" if series.n_runs == 1 else "mean"
        if series.lower is not None:
            description += " +/- SD" if data.uncertainty == "std" else "; min/max"
        label = f"{series.name} ({description}, n={series.n_runs})"
        style = dict(styles.get(series.name, {}))
        if not {"linestyle", "ls"} & set(style):
            style["linestyle"] = "--" if series.name == "Validation" else "-"
        if kind != "training" or len(data.x) == 1:
            style.setdefault("marker", "o")
        (line,) = ax.plot(data.x, series.mean, label=label, **style)
        if series.lower is not None:
            ax.fill_between(
                data.x,
                series.lower,
                series.upper,
                color=line.get_color(),
                alpha=0.18,
                label="_nolegend_",
            )
    ax.set_xlabel(data.x_label)
    ax.set_ylabel(data.metric)
    if legend:
        ax.legend()
    return ax


def plot_learning_curve(
    data: CurveData,
    *,
    ax: Axes,
    legend: bool = True,
    line_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
) -> Axes:
    """Add prepared learning curves to ax and return that identical Axes.

    See plot_training_curve for renderer ownership and styling semantics.
    """
    return _plot(data, "learning", ax, legend, line_kwargs)


def plot_training_curve(
    data: CurveData,
    *,
    ax: Axes,
    legend: bool = True,
    line_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
) -> Axes:
    """Add prepared curves to ax and return that identical Axes.

    Adds lines, optional bands, axis labels, and optionally an Axes legend.
    Existing artists remain. No figure creation, resize, layout, show, save,
    close, or training occurs. The caller owns titles, limits, scales, figure
    lifecycle, and export. ``line_kwargs`` maps series names to Axes.plot keyword
    dictionaries, except label/data/transform. Band colors follow line colors.
    ``legend=False`` leaves an existing legend alone for caller composition.
    """
    return _plot(data, "training", ax, legend, line_kwargs)


def plot_validation_curve(
    data: CurveData,
    *,
    ax: Axes,
    legend: bool = True,
    line_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
) -> Axes:
    """Add prepared validation curves to ax and return that identical Axes.

    See plot_training_curve for renderer ownership and styling semantics.
    """
    return _plot(data, "validation", ax, legend, line_kwargs)
