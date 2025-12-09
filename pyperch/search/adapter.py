
"""
Adapters that connect search strategies to training code.

The adapter pattern keeps the search layer independent of the concrete
training implementation. A TrainerAdapter wraps a configuration builder,
a search strategy, and a training function into a single Optuna compatible
objective callable.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, TypeVar

try:
    from optuna.trial import Trial
except ImportError:  # pragma: no cover
    Trial = Any  # type: ignore

from .builder import TrainConfigBuilder
from .strategy import SearchStrategy

TConfig = TypeVar("TConfig")


class TrainerAdapter:
    """
    Adapter that exposes a training workflow as an Optuna objective.

    The adapter itself does not know about specific trainer classes or
    models. Instead it calls a user supplied training function that
    accepts a configuration object and returns a scalar score.

    This design keeps the integration lightweight and lets callers decide
    how training is performed.
    """

    def __init__(
        self,
        config_builder: TrainConfigBuilder[TConfig],
        strategy: SearchStrategy,
        train_fn: Callable[[TConfig], float],
    ) -> None:
        """
        Parameters
        ----------
        config_builder:
            Builder used to construct configuration instances from
            parameter dictionaries.
        strategy:
            Search strategy that produces parameter dictionaries for
            each trial.
        train_fn:
            Callable that performs training for a given configuration
            and returns a scalar objective value. Higher is considered
            better by default and should match the Optuna study direction.
        """
        self._config_builder = config_builder
        self._strategy = strategy
        self._train_fn = train_fn

    def objective(self, trial: Trial) -> float:
        """
        Optuna compatible objective function.

        This method can be passed directly to `study.optimize`. It will
        use the configured strategy to sample hyperparameters, build a
        training configuration, run the training function, and return
        the resulting score.

        Parameters
        ----------
        trial:
            Optuna trial object.

        Returns
        -------
        float
            Objective value for this trial.
        """
        params = self._strategy.suggest(trial)
        cfg = self._config_builder.with_overrides(params)
        score = self._train_fn(cfg)
        return float(score)
