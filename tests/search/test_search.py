import builtins
import subprocess
import sys

import pytest

from pyperch.search import OptunaSearch


class RecordingTrial:
    def __init__(self):
        self.calls = []

    def suggest_float(self, name, low, high, *, log):
        self.calls.append(("float", name, low, high, log))
        return high

    def suggest_int(self, name, low, high, *, step):
        self.calls.append(("int", name, low, high, step))
        return high

    def suggest_categorical(self, name, choices):
        self.calls.append(("categorical", name, choices))
        return choices[-1]


def test_suggest_params_supports_public_parameter_specs():
    search = OptunaSearch(
        {
            "plain_float": ("float", 0.1, 0.9),
            "log_float": ("float", 0.01, 1.0, True),
            "plain_int": ("int", 1, 5),
            "stepped_int": ("int", 2, 8, 2),
            "choice": ("categorical", ["rhc", "sa"]),
        }
    )
    trial = RecordingTrial()

    params = search.suggest_params(trial)

    assert params == {
        "plain_float": 0.9,
        "log_float": 1.0,
        "plain_int": 5,
        "stepped_int": 8,
        "choice": "sa",
    }
    assert trial.calls == [
        ("float", "plain_float", 0.1, 0.9, False),
        ("float", "log_float", 0.01, 1.0, True),
        ("int", "plain_int", 1, 5, 1),
        ("int", "stepped_int", 2, 8, 2),
        ("categorical", "choice", ["rhc", "sa"]),
    ]


def test_suggest_params_rejects_unsupported_parameter_type():
    search = OptunaSearch({"schedule": ("callable", lambda step: step)})

    with pytest.raises(ValueError, match="Unsupported parameter type: callable"):
        search.suggest_params(RecordingTrial())


def test_search_returns_native_study_and_forwards_trial_and_options():
    optuna = pytest.importorskip("optuna")
    callbacks = []
    search = OptunaSearch(
        {"value": ("categorical", [3, 1, 2])},
        direction="minimize",
        study_kwargs={
            "study_name": "direct-search-test",
            "sampler": optuna.samplers.GridSampler(
                {"value": [3, 1, 2]},
                seed=7,
            ),
        },
    )

    def objective(params, trial):
        assert params == trial.params
        assert isinstance(trial, optuna.trial.Trial)
        return float(params["value"])

    study = search.search(
        objective,
        n_trials=3,
        callbacks=[lambda study, trial: callbacks.append(trial.number)],
    )

    assert isinstance(study, optuna.study.Study)
    assert study.study_name == "direct-search-test"
    assert study.direction is optuna.study.StudyDirection.MINIMIZE
    assert study.best_params == {"value": 1}
    assert study.best_value == 1.0
    assert study.best_trial.number == 1
    assert study.best_trial.number != study.trials[-1].number
    assert sorted(trial.params["value"] for trial in study.trials) == [1, 2, 3]
    assert callbacks == [0, 1, 2]


def test_core_and_search_import_without_optuna():
    script = """
import builtins

import torch

original_import = builtins.__import__


def import_without_optuna(name, *args, **kwargs):
    if name == "optuna" or name.startswith("optuna."):
        raise ImportError("Optuna intentionally unavailable")
    return original_import(name, *args, **kwargs)


builtins.__import__ = import_without_optuna

import pyperch
from pyperch.optim import RHC
from pyperch.search import OptunaSearch

parameter = torch.nn.Parameter(torch.tensor([1.0]))
optimizer = RHC([parameter], step_size=0.1, random_state=7)
optimizer.step(lambda: parameter.square().sum())

assert pyperch.RHC is RHC
assert OptunaSearch({"value": ("int", 1, 1)}).param_space
"""

    subprocess.run([sys.executable, "-c", script], check=True)


def test_search_without_optuna_reports_optional_install_command(monkeypatch):
    original_import = builtins.__import__

    def import_without_optuna(name, *args, **kwargs):
        if name == "optuna":
            raise ImportError("Optuna intentionally unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_optuna)
    search = OptunaSearch({"value": ("int", 1, 1)})

    with pytest.raises(ImportError, match=r"pip install 'pyperch\[optuna\]'"):
        search.search(lambda params, trial: 0.0, n_trials=1)
