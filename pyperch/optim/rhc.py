"""
Randomized Optimization methods for PyPerch.

Based on the original PyPerch optimizers by Jakub Owczarek
(BSD 3-Clause License).

These were also inspired by ABAGAIL’s randomized optimization algorithms - https://github.com/pushkar/ABAGAIL.

Substantial refactoring and redesign by John Mansfield (2026).

"""

from __future__ import annotations

from collections.abc import Callable

import torch

from .base import RandomizedOptimizer


class RHC(RandomizedOptimizer):
    """Randomized Hill Climbing optimizer for arbitrary PyTorch models.

    The closure should return a scalar loss tensor. Lower loss is assumed to be
    better. Gradients are not required.
    """

    def __init__(
        self,
        params,
        step_size: float = 0.1,
        restarts: int = 0,
        restart_interval: int | None = None,
        random_state: int | None = None,
    ):
        if step_size <= 0:
            raise ValueError("step_size must be positive.")
        if restarts < 0:
            raise ValueError("restarts must be >= 0.")
        if restart_interval is not None and restart_interval <= 0:
            raise ValueError("restart_interval must be positive when provided.")

        defaults = {"step_size": step_size}
        super().__init__(params, defaults)

        self.step_size = step_size
        self.restarts = restarts
        self.restart_interval = restart_interval
        self.completed_restarts = 0

        self._generator = torch.Generator()
        if random_state is not None:
            self._generator.manual_seed(random_state)

        self._initialized = False
        self._current_loss: float | None = None
        self._best_params: list[torch.Tensor] | None = None

    def step(self, closure: Callable[[], torch.Tensor]) -> torch.Tensor:
        """Evaluate one candidate move and keep it if loss does not increase."""
        if closure is None:
            raise ValueError("RHC requires a closure that returns the loss.")

        if not self._initialized or self._current_loss is None:
            loss_tensor = self._evaluate(closure)
            loss = float(loss_tensor.detach().item())
            self._initialized = True
            self._current_loss = loss
            self._save_best_if_needed(loss)
            return loss_tensor

        old_params = self._clone_params()
        old_loss = self._current_loss

        self._propose_step()
        self.proposed_steps += 1

        candidate_loss_tensor = self._evaluate(closure)
        candidate_loss = float(candidate_loss_tensor.detach().item())

        if candidate_loss <= old_loss:
            self._current_loss = candidate_loss
            self.accepted_steps += 1
            self._save_best_if_needed(candidate_loss)
            return candidate_loss_tensor

        self._restore_params(old_params)
        self.rejected_steps += 1
        self._maybe_restart()
        return torch.tensor(
            old_loss,
            dtype=candidate_loss_tensor.dtype,
            device=candidate_loss_tensor.device,
        )

    def _evaluate(self, closure: Callable[[], torch.Tensor]) -> torch.Tensor:
        """Run the closure and count one objective evaluation."""
        with torch.enable_grad():
            loss_tensor = closure()
        loss = float(loss_tensor.detach().item())
        self._record_eval(loss)
        return loss_tensor

    @torch.no_grad()
    def _propose_step(self) -> None:
        """Add Gaussian noise to each trainable parameter."""
        for group in self.param_groups:
            step_size = group["step_size"]
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                noise = torch.randn(
                    p.shape,
                    generator=self._generator,
                    device=p.device,
                    dtype=p.dtype,
                )
                p.add_(step_size * noise)

    @torch.no_grad()
    def _maybe_restart(self) -> None:
        """Randomly reset parameters when the restart schedule is reached."""
        if self.restart_interval is None:
            return
        if self.completed_restarts >= self.restarts:
            return
        if self.proposed_steps % self.restart_interval != 0:
            return

        for p in self._parameters():
            noise = torch.randn(
                p.shape,
                generator=self._generator,
                device=p.device,
                dtype=p.dtype,
            )
            p.copy_(noise)

        self.completed_restarts += 1
        self._current_loss = None

    @torch.no_grad()
    def _save_best_if_needed(self, loss: float) -> None:
        """Save the current parameters when they improve the best loss."""
        if self.best_loss is None or loss <= self.best_loss:
            self.best_loss = loss
            self._best_params = self._clone_params()

    @torch.no_grad()
    def restore_best(self) -> None:
        """Restore the best parameter values observed so far."""
        if self._best_params is not None:
            self._restore_params(self._best_params)
