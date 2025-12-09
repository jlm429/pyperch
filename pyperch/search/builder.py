
"""
Configuration builder utilities.

These helpers take a base configuration object and apply a set of
parameter overrides produced by a search strategy. The builder is
designed to work with simple dataclasses as well as pydantic models.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Generic, TypeVar

TConfig = TypeVar("TConfig")


class TrainConfigBuilder(Generic[TConfig]):
    """
    Helper for building training configuration objects with overrides.

    This builder starts from a base configuration instance and applies
    a dictionary of updates. Keys may refer to top level attributes or
    use dotted paths to update nested attributes.

    The exact config type is generic so that it can be used with
    dataclasses, pydantic models, or other simple containers.

    Examples
    --------
    >>> base = TrainConfig(max_epochs=20)
    >>> builder = TrainConfigBuilder(base)
    >>> cfg = builder.with_overrides({"max_epochs": 50})
    """

    def __init__(self, base_config: TConfig) -> None:
        self._base_config = base_config

    @property
    def base_config(self) -> TConfig:
        """
        Return the base configuration used by this builder.
        """
        return self._base_config

    def with_overrides(self, params: Dict[str, Any]) -> TConfig:
        """
        Return a new configuration instance with the provided overrides.

        Parameters
        ----------
        params:
            Mapping from parameter names to values. Keys may use dotted
            paths, for example "optimizer_config.step_size".

        Returns
        -------
        TConfig
            New configuration instance with overrides applied.
        """
        cfg = copy.deepcopy(self._base_config)
        for key, value in params.items():
            self._apply_single_override(cfg, key, value)
        return cfg

    def _apply_single_override(self, cfg: Any, key: str, value: Any) -> None:
        parts = key.split(".")
        target = cfg
        for name in parts[:-1]:
            if not hasattr(target, name):
                raise AttributeError(
                    f"Cannot apply override for path '{key}'. "
                    f"Missing attribute '{name}' on object of type {type(target)!r}."
                )
            target = getattr(target, name)
        leaf_name = parts[-1]
        if not hasattr(target, leaf_name):
            raise AttributeError(
                f"Cannot apply override for path '{key}'. "
                f"Missing attribute '{leaf_name}' on object of type {type(target)!r}."
            )
        setattr(target, leaf_name, value)
