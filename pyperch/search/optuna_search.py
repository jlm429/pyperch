
"""
Thin wrapper around Optuna studies.

The OptunaSearch class provides a small convenience layer for running
hyperparameter optimization using Optuna together with a TrainerAdapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

try:
    import optuna
    from optuna.study import Study
except ImportError:  # pragma: no cover
    optuna = None  # type: ignore
    Study = Any  # type: ignore

from .adapter import TrainerAdapter


@dataclass
class OptunaSearch:
    """
    Convenience wrapper around an Optuna study.

    This class is intentionally small. It delegates trial evaluation to a
    TrainerAdapter and exposes only a few helper methods so that 
    users can still work with the underlying Optuna study directly if they
    prefer.
    """

    study: Study
    adapter: TrainerAdapter

    def run(
        self,
        n_trials: int,
        n_jobs: int = 1,
        timeout: Optional[float] = None,
        gc_after_trial: bool = True,
    ) -> Study:
        """
        Run hyperparameter optimization.

        Parameters
        ----------
        n_trials:
            Maximum number of trials to run.
        n_jobs:
            Number of parallel workers. Set to a value greater than one
            to enable parallel evaluation of trials.
        timeout:
            Maximum optimization time in seconds. If None there is no
            time limit.
        gc_after_trial:
            Whether to trigger garbage collection after each trial.

        Returns
        -------
        Study
            The underlying Optuna study instance with completed trials.
        """
        if optuna is None:
            raise RuntimeError(
                "Optuna is not installed. Install optuna to use OptunaSearch."
            )

        self.study.optimize(
            self.adapter.objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            timeout=timeout,
            gc_after_trial=gc_after_trial,
        )
        return self.study

    @property
    def best_params(self) -> dict:
        """
        Return the best hyperparameters found so far.
        """
        return dict(self.study.best_params)

    @property
    def best_value(self) -> float:
        """
        Return the best objective value found so far.
        """
        return float(self.study.best_value)

    @property
    def best_trial(self) -> Any:
        """
        Return the best trial object.
        """
        return self.study.best_trial

    @classmethod
        # type: ignore[override]
    def from_sqlite(
        cls,
        adapter: TrainerAdapter,
        study_name: str = "pyperch_optuna",
        storage: str = "sqlite:///optuna_study.db",
        direction: str = "maximize",
        load_if_exists: bool = True,
    ) -> "OptunaSearch":
        """
        Create an OptunaSearch instance backed by a SQLite database.

        This helper is a convenient starting point for typical projects.
        Using SQLite provides a persistent study that supports parallel
        workers without requiring a separate database server.

        Parameters
        ----------
        adapter:
            TrainerAdapter used to evaluate trials.
        study_name:
            Name of the Optuna study.
        storage:
            Storage URL. By default this points to a local SQLite file.
        direction:
            Optimization direction. Either "maximize" or "minimize".
        load_if_exists:
            Whether to reuse an existing study with the same name.

        Returns
        -------
        OptunaSearch
            New search wrapper instance.
        """
        if optuna is None:
            raise RuntimeError(
                "Optuna is not installed. Install optuna to use OptunaSearch."
            )

        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction=direction,
            load_if_exists=load_if_exists,
        )
        return cls(study=study, adapter=adapter)
