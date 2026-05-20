from __future__ import annotations

from collections.abc import Iterable

import torch


class RandomizedOptimizer(torch.optim.Optimizer):
    """Base class for randomized optimizers that operate on PyTorch parameters."""

    def __init__(self, params: Iterable[torch.nn.Parameter], defaults: dict):
        super().__init__(params, defaults)
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
    def _clone_params(self) -> list[torch.Tensor]:
        """Copy the current trainable parameter values."""
        return [p.detach().clone() for p in self._parameters()]

    @torch.no_grad()
    def _restore_params(self, values: list[torch.Tensor]) -> None:
        """Restore trainable parameters from a copied parameter list."""
        for p, value in zip(self._parameters(), values):
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
