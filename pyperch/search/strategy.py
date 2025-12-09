
"""
Search strategy interfaces and Optuna based implementations to separate the definition of the search
space from the mechanics of running a training loop. A strategy produces
a dictionary of hyperparameters for a given trial.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Protocol

try:
    import optuna
    from optuna.trial import Trial
except ImportError:  # pragma: no cover
    optuna = None  # type: ignore
    Trial = Any  # type: ignore


class SearchStrategy(Protocol):
    """
    Interface for hyperparameter search strategies.

    Responsible for turning a trial object into a mapping
    from parameter names to concrete values. The meaning of the keys is
    up to the caller. For example, keys might correspond to fields on a
    training configuration object or nested attributes such as
    "optimizer_config.step_size".
    """

    def suggest(self, trial: Trial) -> Dict[str, Any]:
        """
        Produce a dictionary of hyperparameters for the given trial.

        Parameters
        ----------
        trial:
            The underlying trial object. For Optuna based strategies this
            will be an optuna.trial.Trial instance.

        Returns
        -------
        Dict[str, Any]
            A mapping from parameter names to values.
        """
        raise NotImplementedError


class OptunaStrategy:
    """
    Search strategy that delegates to a user supplied suggestion function.

    This class is a thin wrapper around a callable that takes an Optuna
    trial and returns a parameter dictionary. It allows users to
    define search spaces while keeping the search
    layer decoupled from the core training code.

    Examples
    --------
    >>> def suggest(trial):
    ...     return {
    ...         "optimizer_config.step_size": trial.suggest_float("step_size", 0.01, 0.2),
    ...         "max_epochs": trial.suggest_int("max_epochs", 10, 50),
    ...     }
    >>>
    >>> strategy = OptunaStrategy(suggest)
    """

    def __init__(self, suggest_fn: Callable[[Trial], Dict[str, Any]]) -> None:
        if optuna is None:
            raise RuntimeError(
                "Optuna is not installed. Install optuna to use OptunaStrategy."
            )
        self._suggest_fn = suggest_fn

    def suggest(self, trial: Trial) -> Dict[str, Any]:
        """
        Call the wrapped suggestion function.

        Parameters
        ----------
        trial:
            Optuna trial object.

        Returns
        -------
        Dict[str, Any]
            Suggested hyperparameters.
        """
        return self._suggest_fn(trial)
