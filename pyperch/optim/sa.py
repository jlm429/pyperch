"""Simulated annealing optimizer.

Randomized Optimization methods for PyPerch.

Based on the original PyPerch optimizers by Jakub Owczarek
(BSD 3-Clause License).

These were also inspired by ABAGAIL’s randomized optimization algorithms - https://github.com/pushkar/ABAGAIL.

Substantial refactoring and redesign by John Mansfield (2026).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from .base import RandomizedOptimizer


class SA(RandomizedOptimizer):
    """Simulated annealing optimizer for arbitrary PyTorch models.

    This optimizer uses random perturbations instead of gradients.

    Expected usage:

        optimizer = SA(model.parameters(), step_size=0.1)

        def closure():
            output = model(X)
            loss = loss_fn(output, y)
            return loss

        loss = optimizer.step(closure)

    Lower loss is assumed to be better.
    """

    def __init__(
        self,
        params,
        step_size: float = 0.1,
        temperature: float = 1.0,
        min_temperature: float = 0.1,
        cooling: float = 0.95,
        random_state: int | None = None,
    ):
        if step_size <= 0:
            raise ValueError("step_size must be positive.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if min_temperature <= 0:
            raise ValueError("min_temperature must be positive.")
        if cooling <= 0 or cooling > 1:
            raise ValueError("cooling must be in the interval (0, 1].")

        defaults = {
            "step_size": step_size,
        }

        super().__init__(params, defaults)

        self.step_size = step_size
        self.temperature = temperature
        self.min_temperature = min_temperature
        self.cooling = cooling

        self._generator = torch.Generator()
        if random_state is not None:
            self._generator.manual_seed(random_state)

        self._initialized = False
        self._current_loss: float | None = None
        self._best_params: list[torch.Tensor] | None = None

    def step(self, closure: Callable[[], torch.Tensor]) -> torch.Tensor:
        if closure is None:
            raise ValueError("SA requires a closure that returns the loss.")

        if not self._initialized:
            with torch.enable_grad():
                loss_tensor = closure()

            loss = float(loss_tensor.detach().item())

            self.function_evals += 1
            self._initialized = True
            self._current_loss = loss
            self._update_best_loss(loss)
            self._best_params = self._clone_params()

            return loss_tensor

        trainable = self._parameters()

        if not trainable:
            with torch.enable_grad():
                loss_tensor = closure()

            loss = float(loss_tensor.detach().item())
            self.function_evals += 1
            self._current_loss = loss
            self._update_best_loss(loss)

            return loss_tensor

        param_index = torch.randint(
            low=0,
            high=len(trainable),
            size=(1,),
            generator=self._generator,
        ).item()

        param = trainable[param_index]

        old_param = param.detach().clone()

        with torch.no_grad():
            noise = (
                torch.rand(
                    param.shape,
                    generator=self._generator,
                    device=param.device,
                    dtype=param.dtype,
                )
                - 0.5
            )

            param.add_(self.step_size * noise)

        self.proposed_steps += 1

        with torch.enable_grad():
            candidate_loss_tensor = closure()

        candidate_loss = float(candidate_loss_tensor.detach().item())
        self.function_evals += 1

        delta = candidate_loss - self._current_loss

        accept = False

        if delta <= 0:
            accept = True
        else:
            acceptance_probability = math.exp(-delta / max(self.temperature, 1e-12))

            random_value = torch.rand(
                size=(1,),
                generator=self._generator,
            ).item()

            accept = random_value < acceptance_probability

        if accept:
            self.accepted_steps += 1
            self._current_loss = candidate_loss

            if self.best_loss is None or candidate_loss < self.best_loss:
                self.best_loss = candidate_loss
                self._best_params = self._clone_params()

            result = candidate_loss_tensor

        else:
            with torch.no_grad():
                param.copy_(old_param)

            self.rejected_steps += 1

            result = torch.tensor(
                self._current_loss,
                dtype=candidate_loss_tensor.dtype,
                device=candidate_loss_tensor.device,
            )

        self.temperature = max(
            self.temperature * self.cooling,
            self.min_temperature,
        )

        return result

    @torch.no_grad()
    def restore_best(self) -> None:
        """Restore the best parameters observed so far."""
        if self._best_params is not None:
            self._restore_params(self._best_params)
