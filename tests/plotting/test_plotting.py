import builtins
import subprocess
import sys

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pyperch.plotting import (  # noqa: E402
    plot_learning_curve,
    plot_training_curve,
    plot_validation_curve,
    prepare_learning_curve,
    prepare_training_curve,
    prepare_validation_curve,
)


@pytest.fixture
def axes():
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    yield fig, axes
    plt.close(fig)


def prepared(kind, **kwargs):
    values = [[1, 3, 5], [2, 4, 6]]
    if kind == "learning":
        return prepare_learning_curve([10, 20], values, values, metric="MSE", **kwargs)
    if kind == "validation":
        return prepare_validation_curve(
            [2, 8], values, values, param_name="Width", metric="MSE", **kwargs
        )
    return prepare_training_curve([0, 1], {"RHC": values}, metric="MSE", **kwargs)


RENDERERS = {
    "learning": plot_learning_curve,
    "training": plot_training_curve,
    "validation": plot_validation_curve,
}


@pytest.mark.parametrize("kind", RENDERERS)
def test_axes_identity_composition_and_no_figure_side_effects(kind, axes, monkeypatch):
    fig, (ax, other) = axes
    (existing,) = ax.plot([0, 1], [2, 2], label="reference")
    other.set_title("untouched")
    ax.set_title("Caller title")
    ax.set_xlim(-1, 30)
    ax.set_ylim(-2, 10)
    ax.set_yscale("symlog")
    initial_size = fig.get_size_inches().copy()
    positions = [a.get_position().bounds for a in fig.axes]
    fignums = plt.get_fignums()
    plt.sca(other)

    def forbidden(*args, **kwargs):
        pytest.fail("Renderer attempted a forbidden figure operation")

    with monkeypatch.context() as patch:
        for name in ("show", "savefig", "close", "figure", "subplots", "tight_layout"):
            patch.setattr(plt, name, forbidden)
        for name in (
            "show",
            "savefig",
            "set_size_inches",
            "tight_layout",
            "set_layout_engine",
            "set_constrained_layout",
            "subplots_adjust",
            "add_subplot",
            "add_axes",
            "clear",
        ):
            patch.setattr(fig, name, forbidden)
        result = RENDERERS[kind](prepared(kind), ax=ax)
    assert result is ax
    assert ax.lines[0] is existing
    assert plt.gca() is other
    assert len(other.lines) == 0
    assert other.get_title() == "untouched"
    assert ax.get_title() == "Caller title"
    assert ax.get_xlim() == (-1, 30)
    assert ax.get_ylim() == (-2, 10)
    assert ax.get_yscale() == "symlog"
    assert [a.get_position().bounds for a in fig.axes] == positions
    np.testing.assert_array_equal(fig.get_size_inches(), initial_size)
    assert plt.get_fignums() == fignums
    assert ax.get_ylabel() == "MSE"
    assert "reference" in ax.get_legend_handles_labels()[1]
    assert all("n=3" in line.get_label() for line in ax.lines[1:])
    fig.canvas.draw()


@pytest.mark.parametrize("kind", RENDERERS)
def test_explicit_axes_and_matching_preparation_required(kind, axes):
    data = prepared(kind)
    with pytest.raises(TypeError):
        RENDERERS[kind](data)
    with pytest.raises(TypeError, match="explicit"):
        RENDERERS[kind](data, ax=None)
    with pytest.raises(ValueError, match="Expected prepared"):
        RENDERERS[kind](object(), ax=axes[1][0])
    wrong = prepared("training" if kind != "training" else "learning")
    with pytest.raises(ValueError, match="Expected prepared"):
        RENDERERS[kind](wrong, ax=axes[1][0])


@pytest.mark.parametrize(
    "uncertainty, bounds",
    [("std", ((1, 2), (5, 6))), ("range", ((1, 2), (5, 6))), (None, (None, None))],
)
def test_repeat_aggregation(uncertainty, bounds):
    data = prepared("training", uncertainty=uncertainty)
    series = data.series[0]
    assert series.n_runs == 3
    assert series.mean == (3, 4)
    assert (series.lower, series.upper) == bounds


def test_std_is_sample_sd_and_band_is_not_clipped(axes):
    data = prepare_training_curve([0], {"Accuracy": [[0.0, 1.0]]}, metric="Accuracy")
    series = data.series[0]
    np.testing.assert_allclose(series.lower, [0.5 - np.sqrt(0.5)])
    np.testing.assert_allclose(series.upper, [0.5 + np.sqrt(0.5)])
    ax = plot_training_curve(data, ax=axes[1][0])
    band_y = ax.collections[0].get_paths()[0].vertices[:, 1]
    assert band_y.min() < 0 and band_y.max() > 1
    assert "SD" in ax.lines[0].get_label()
    assert ax.lines[0].get_marker() == "o"


@pytest.mark.parametrize("uncertainty", [None, "std", "range"])
def test_single_run_has_no_invented_uncertainty(axes, uncertainty):
    data = prepare_training_curve(
        [0, 1], {"loss": [4, 2]}, metric="MSE", uncertainty=uncertainty
    )
    assert data.series[0].lower is None and data.series[0].upper is None
    ax = plot_training_curve(data, ax=axes[1][0])
    assert not ax.collections
    assert "single run, n=1" in ax.lines[0].get_label()


def test_inputs_are_copied_and_series_can_have_different_repeat_counts():
    source = np.array([[1, 3], [2, 4]])
    data = prepare_training_curve([0, 1], {"A": source, "B": [8, 7]}, metric="Loss")
    source[:] = 99
    assert data.series[0].mean == (2, 3)
    assert [s.n_runs for s in data.series] == [2, 1]


def test_styles_legend_and_overlay(axes):
    ax = axes[1][0]
    ax.plot([0, 1], [0, 0], label="baseline")
    legend = ax.legend()
    data = prepared("training")
    plot_training_curve(
        data, ax=ax, legend=False, line_kwargs={"RHC": {"color": "red", "ls": ":"}}
    )
    assert ax.get_legend() is legend
    assert ax.lines[-1].get_color() == "red"
    assert ax.lines[-1].get_linestyle() == ":"
    plot_training_curve(data, ax=ax)
    assert len(ax.lines) == 3
    assert len(ax.collections) == 2


def test_categorical_validation_and_numeric_log_scale(axes):
    data = prepare_validation_curve(
        ["small", "large"],
        [0.5, 0.9],
        [0.6, 0.7],
        param_name="Architecture",
        metric="Accuracy",
    )
    ax = plot_validation_curve(data, ax=axes[1][0])
    axes[0].canvas.draw()
    assert [t.get_text() for t in ax.get_xticklabels()] == ["small", "large"]
    assert ax.get_xlabel() == "Architecture"
    other = axes[1][1]
    other.set_xscale("log")
    plot_validation_curve(prepared("validation"), ax=other)
    assert other.get_xscale() == "log"


@pytest.mark.parametrize(
    "x",
    [
        [],
        [[1, 2]],
        [1, 1],
        [2, 1],
        [1, np.inf],
        [-1, 2],
        [1, np.nan],
        [True, False],
        [1 + 1j, 2],
    ],
)
def test_bad_training_coordinates(x):
    with pytest.raises(ValueError):
        prepare_training_curve(x, {"a": [1, 2]}, metric="loss")


@pytest.mark.parametrize(
    "values",
    [
        [],
        [[1], [2], [3]],
        [[1, 2], [3]],
        np.zeros((2, 0)),
        np.zeros((2, 2, 1)),
        [np.nan, 1],
        [np.inf, 1],
        ["1", "2"],
        [1 + 2j, 3],
        [True, False],
        [[1e308, 1e308], [1e308, 1e308]],
    ],
)
def test_bad_scores(values):
    with pytest.raises(ValueError):
        prepare_training_curve([0, 1], {"a": values}, metric="loss")


@pytest.mark.parametrize("sizes", [[0, 10], [1.5, 2], [20, 10]])
def test_learning_sizes_are_sample_counts(sizes):
    with pytest.raises(ValueError):
        prepare_learning_curve(sizes, [1, 2], [2, 3], metric="MSE")


@pytest.mark.parametrize(
    "values", [["x", "x"], ["", "x"], ["x", None], ["small", 2], [1, "large"]]
)
def test_invalid_categories(values):
    with pytest.raises(ValueError):
        prepare_validation_curve(values, [1, 2], [2, 3], param_name="p", metric="MSE")


def test_paired_repeats_required():
    with pytest.raises(ValueError, match="same number"):
        prepare_learning_curve([5, 10], [[1, 2], [3, 4]], [1, 2], metric="MSE")


@pytest.mark.parametrize(
    "kwargs", [{"metric": ""}, {"x_label": ""}, {"uncertainty": "ci"}]
)
def test_invalid_metadata(kwargs):
    with pytest.raises(ValueError):
        prepare_training_curve([0], {"a": [1]}, **({"metric": "MSE"} | kwargs))


@pytest.mark.parametrize("scores", [{}, [], {"": [1]}, {"_hidden": [1]}])
def test_invalid_series(scores):
    with pytest.raises(ValueError):
        prepare_training_curve([0], scores, metric="MSE")


@pytest.mark.parametrize(
    "style",
    [
        {"unknown": {}},
        {"RHC": {"label": "false"}},
        {"RHC": {"transform": None}},
        {"RHC": 5},
    ],
)
def test_invalid_styles_do_not_add_artists(axes, style):
    with pytest.raises(ValueError):
        plot_training_curve(prepared("training"), ax=axes[1][0], line_kwargs=style)
    assert not axes[1][0].lines


def test_preparation_needs_no_optional_packages_or_training():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from importlib.abc import MetaPathFinder
class Block(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'matplotlib', 'optuna', 'sklearn'}:
            raise ImportError('optional dependency blocked')
sys.meta_path.insert(0, Block())
import pyperch
from pyperch.optim import RHC, SA, GA
from pyperch.plotting import prepare_learning_curve, prepare_training_curve
from pyperch.plotting import prepare_validation_curve

def fail(*args, **kwargs):
    raise AssertionError('training attempted')
for optimizer in (RHC, SA, GA):
    optimizer.step = fail
assert prepare_learning_curve([2], [1], [2], metric='MSE').series[0].mean == (1,)
assert prepare_training_curve([0], {'a': [1]}, metric='MSE').series[0].mean == (1,)
assert prepare_validation_curve([2], [1], [2], param_name='p', metric='MSE')
assert 'matplotlib' not in sys.modules
""",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_missing_matplotlib_install_message(monkeypatch):
    original = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "matplotlib.axes":
            raise ImportError("blocked")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match=r"pyperch\[plotting\]"):
        plot_training_curve(prepared("training"), ax=None)


def test_observed_range_is_not_standard_deviation(axes):
    data = prepare_training_curve(
        [0, 1],
        {"Loss": [[0, 0, 9], [1, 1, 10]]},
        metric="MSE",
        uncertainty="range",
    )
    series = data.series[0]
    assert series.mean == (3, 4)
    assert series.lower == (0, 1)
    assert series.upper == (9, 10)
    ax = plot_training_curve(data, ax=axes[1][0])
    vertices = ax.collections[0].get_paths()[0].vertices
    assert vertices[:, 1].min() == 0
    assert vertices[:, 1].max() == 10
    assert "min/max" in ax.lines[0].get_label()
