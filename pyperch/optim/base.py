from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch


class RandomizedOptimizer(torch.optim.Optimizer):
    """Base class for randomized optimizers that operate on PyTorch parameters."""

    _checkpoint_state_key = "_pyperch"
    _checkpoint_version = 1
    _group_options: frozenset[str] = frozenset()
    _optimizer_level_options: frozenset[str] = frozenset({"random_state"})
    _known_options = frozenset(
        {
            "cooling",
            "initialization_step_size",
            "min_temperature",
            "mutation_rate",
            "mutation_step_size",
            "population_size",
            "random_state",
            "restart_interval",
            "restart_scale",
            "restarts",
            "step_size",
            "temperature",
        }
    )

    def __init__(self, params: Iterable[torch.nn.Parameter], defaults: dict):
        self._group_lifecycle_ready = False
        super().__init__(params, defaults)
        self.reset_counters()
        self.register_state_dict_post_hook(
            self._add_checkpoint_state,
            prepend=True,
        )
        self.register_load_state_dict_pre_hook(
            self._prepare_checkpoint_load,
            prepend=True,
        )
        self.register_load_state_dict_post_hook(
            self._restore_checkpoint_state,
            prepend=True,
        )
        self._pending_legacy_state: dict[str, Any] | None = None
        self._group_lifecycle_ready = True

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        """Add a group and start fresh bookkeeping for the enlarged search space."""
        if isinstance(param_group, dict):
            self._validate_param_group(param_group)

        super().add_param_group(param_group)

        if self._group_lifecycle_ready:
            self.reset_counters()

    def reset_counters(self) -> None:
        """Reset optimization counters without changing model parameters."""
        self.function_evals = 0
        self.proposed_steps = 0
        self.accepted_steps = 0
        self.rejected_steps = 0
        self.best_loss: float | None = None

    @torch.no_grad()
    def _parameters(self) -> list[torch.nn.Parameter]:
        """Return trainable parameters managed by this optimizer."""
        return [
            p for group in self.param_groups for p in group["params"] if p.requires_grad
        ]

    @torch.no_grad()
    def _parameters_with_groups(
        self,
    ) -> list[tuple[torch.nn.Parameter, dict[str, Any]]]:
        """Return trainable parameters paired with their parameter groups."""
        return [
            (p, group)
            for group in self.param_groups
            for p in group["params"]
            if p.requires_grad
        ]

    @torch.no_grad()
    def _all_parameters(self) -> list[torch.nn.Parameter]:
        """Return every parameter managed by this optimizer in stable group order."""
        return [p for group in self.param_groups for p in group["params"]]

    @torch.no_grad()
    def _clone_params(self) -> list[torch.Tensor]:
        """Copy the current trainable parameter values."""
        return [p.detach().clone() for p in self._parameters()]

    @torch.no_grad()
    def _restore_params(self, values: list[torch.Tensor]) -> None:
        """Restore trainable parameters from a copied parameter list."""
        for p, value in zip(self._parameters(), values):
            p.copy_(value)

    @torch.no_grad()
    def _clone_all_params(self) -> list[torch.Tensor]:
        """Copy every managed parameter for a complete best-model checkpoint."""
        return [p.detach().clone() for p in self._all_parameters()]

    @torch.no_grad()
    def _restore_all_params(self, values: list[torch.Tensor]) -> None:
        """Restore every managed parameter from a complete checkpoint."""
        for p, value in zip(self._all_parameters(), values):
            p.copy_(value)

    def _record_eval(self, loss: float | None = None) -> None:
        """Record one objective evaluation and optionally update the best loss."""
        self.function_evals += 1

        if loss is not None:
            self._update_best_loss(loss)

    def _update_best_loss(self, loss: float) -> None:
        """Update the best loss when improved."""
        if self.best_loss is None or loss < self.best_loss:
            self.best_loss = loss

    @staticmethod
    def _make_generator(random_state: int | None) -> torch.Generator:
        """Create a private CPU generator with optional reproducible seeding."""
        generator = torch.Generator()
        if random_state is None:
            generator.seed()
        else:
            generator.manual_seed(random_state)
        return generator

    def _rand_like(self, reference: torch.Tensor) -> torch.Tensor:
        """Sample uniformly with the private CPU generator and preserve metadata."""
        return torch.rand(
            reference.shape,
            generator=self._generator,
            dtype=reference.dtype,
        ).to(reference.device)

    def _randn_like(self, reference: torch.Tensor) -> torch.Tensor:
        """Sample normally with the private CPU generator and preserve metadata."""
        return torch.randn(
            reference.shape,
            generator=self._generator,
            dtype=reference.dtype,
        ).to(reference.device)

    def _validate_param_group(self, param_group: dict[str, Any]) -> None:
        optimizer_level = sorted(
            self._optimizer_level_options.intersection(param_group)
        )
        if optimizer_level:
            names = ", ".join(optimizer_level)
            raise ValueError(
                f"{names} must be configured on the optimizer, not in a parameter "
                "group."
            )

        unsupported = sorted(
            (self._known_options - self._group_options).intersection(param_group)
        )
        if unsupported:
            names = ", ".join(unsupported)
            raise ValueError(
                f"{names} is not a parameter-group option for {type(self).__name__}."
            )

        self._validate_group_options(param_group)

    def _validate_group_options(self, param_group: dict[str, Any]) -> None:
        """Validate algorithm-specific options explicitly present in a group."""

    def _algorithm_checkpoint_state(self) -> dict[str, Any]:
        """Return algorithm-specific optimizer-level and run state."""
        return {}

    def _load_algorithm_checkpoint_state(self, state: dict[str, Any]) -> None:
        """Restore algorithm-specific optimizer-level and run state."""
        if state:
            raise ValueError("Checkpoint contains unexpected algorithm state.")

    def _migrate_legacy_param_groups(
        self,
        param_groups: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Remove legacy ineffective group options and recover global settings."""
        return None

    def _load_legacy_algorithm_state(self, state: dict[str, Any] | None) -> None:
        """Restore global settings recoverable from a pre-checkpoint state dict."""

    def _checkpoint_state(self) -> dict[str, Any]:
        best_params = None
        if self._best_params is not None:
            best_params = [value.detach().clone() for value in self._best_params]

        return {
            "version": self._checkpoint_version,
            "optimizer": type(self).__name__,
            "group_defaults": {
                name: self.defaults[name]
                for name in self._group_options
                if name in self.defaults
            },
            "generator_state": self._generator.get_state().clone(),
            "function_evals": self.function_evals,
            "proposed_steps": self.proposed_steps,
            "accepted_steps": self.accepted_steps,
            "rejected_steps": self.rejected_steps,
            "best_loss": self.best_loss,
            "initialized": self._initialized,
            "current_loss": self._current_loss,
            "best_params": best_params,
            "algorithm": self._algorithm_checkpoint_state(),
        }

    def _add_checkpoint_state(
        self,
        optimizer: RandomizedOptimizer,
        state_dict: dict[str, Any],
    ) -> dict[str, Any]:
        state_dict["state"][self._checkpoint_state_key] = self._checkpoint_state()
        return state_dict

    def _prepare_checkpoint_load(
        self,
        optimizer: RandomizedOptimizer,
        state_dict: dict[str, Any],
    ) -> dict[str, Any]:
        prepared = state_dict.copy()
        prepared["param_groups"] = [
            group.copy() for group in state_dict["param_groups"]
        ]
        checkpoint = state_dict["state"].get(self._checkpoint_state_key)

        if checkpoint is None:
            self._pending_legacy_state = self._migrate_legacy_param_groups(
                prepared["param_groups"]
            )
        else:
            self._pending_legacy_state = None
            if checkpoint.get("version") != self._checkpoint_version:
                raise ValueError("Unsupported PyPerch optimizer checkpoint version.")
            if checkpoint.get("optimizer") != type(self).__name__:
                raise ValueError(
                    "Cannot load a checkpoint created by a different PyPerch optimizer."
                )

        for group in prepared["param_groups"]:
            self._validate_param_group(group)

        return prepared

    def _restore_checkpoint_state(self, optimizer: RandomizedOptimizer) -> None:
        checkpoint = self.state.pop(self._checkpoint_state_key, None)
        if checkpoint is None:
            self.reset_counters()
            self._load_legacy_algorithm_state(self._pending_legacy_state)
            self._pending_legacy_state = None
            return

        group_defaults = checkpoint["group_defaults"]
        self._validate_group_options(group_defaults)
        self.defaults.update(group_defaults)

        saved_best = checkpoint["best_params"]
        if saved_best is None:
            best_params = None
        else:
            parameters = self._all_parameters()
            if len(saved_best) != len(parameters):
                raise ValueError(
                    "Checkpoint best parameters do not match the optimizer's "
                    "parameter count."
                )
            best_params = [
                value.detach()
                .to(device=parameter.device, dtype=parameter.dtype)
                .clone()
                for parameter, value in zip(parameters, saved_best)
            ]

        self._generator.set_state(checkpoint["generator_state"].detach().cpu())
        self.function_evals = checkpoint["function_evals"]
        self.proposed_steps = checkpoint["proposed_steps"]
        self.accepted_steps = checkpoint["accepted_steps"]
        self.rejected_steps = checkpoint["rejected_steps"]
        self.best_loss = checkpoint["best_loss"]
        self._initialized = checkpoint["initialized"]
        self._current_loss = checkpoint["current_loss"]
        self._best_params = best_params
        self._load_algorithm_checkpoint_state(checkpoint["algorithm"])
