
"""
Factory helpers for constructing search objects.

The factory module provides small convenience functions and classes that
set up the pieces required to run hyperparameter search.
"""

from __future__ import annotations

from typing import Any

from .adapter import TrainerAdapter
from .optuna_search import OptunaSearch


class SearchFactory:
    """
    Factory for creating search objects.

    At present this focuses on Optuna based search but the same pattern
    can be extended to support additional backends in the future.
    """

    @staticmethod
    def optuna_sqlite(
        adapter: TrainerAdapter,
        study_name: str = "pyperch_optuna",
        storage: str = "sqlite:///optuna_study.db",
        direction: str = "maximize",
        load_if_exists: bool = True,
    ) -> OptunaSearch:
        """
        Create an OptunaSearch instance backed by SQLite.

        Parameters
        ----------
        adapter:
            TrainerAdapter used to evaluate trials.
        study_name:
            Name of the Optuna study.
        storage:
            Storage URL. Uses a local SQLite file by default.
        direction:
            Optimization direction. Either "maximize" or "minimize".
        load_if_exists:
            Whether to reuse an existing study if it already exists.

        Returns
        -------
        OptunaSearch
            Configured search object.
        """
        return OptunaSearch.from_sqlite(
            adapter=adapter,
            study_name=study_name,
            storage=storage,
            direction=direction,
            load_if_exists=load_if_exists,
        )
