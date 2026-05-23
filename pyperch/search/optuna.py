from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

ParamSpec = tuple[Any, ...]
ParamSpace = dict[str, ParamSpec]


@dataclass
class OptunaSearch:
    param_space: ParamSpace
    direction: str = "minimize"
    study_kwargs: dict[str, Any] | None = None

    def suggest_params(self, trial) -> dict[str, Any]:
        params = {}

        for name, spec in self.param_space.items():
            kind = spec[0]

            if kind == "float":
                _, low, high, *options = spec
                log = bool(options[0]) if options else False
                params[name] = trial.suggest_float(name, low, high, log=log)

            elif kind == "int":
                _, low, high, *options = spec
                step = int(options[0]) if options else 1
                params[name] = trial.suggest_int(name, low, high, step=step)

            elif kind == "categorical":
                _, choices = spec
                params[name] = trial.suggest_categorical(name, choices)

            else:
                raise ValueError(f"Unsupported parameter type: {kind}")

        return params

    def search(
        self,
        objective_fn: Callable[[dict[str, Any], Any], float],
        n_trials: int = 20,
        **optimize_kwargs,
    ):
        try:
            import optuna
        except ImportError as exc:
            raise ImportError(
                "Optuna support requires the optional dependency. "
                "Install with: poetry install --extras optuna"
            ) from exc

        study = optuna.create_study(
            direction=self.direction,
            **(self.study_kwargs or {}),
        )

        def objective(trial):
            params = self.suggest_params(trial)
            return objective_fn(params, trial)

        study.optimize(objective, n_trials=n_trials, **optimize_kwargs)
        return study