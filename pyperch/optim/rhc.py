"""
Author: Jakub Owczarek 
BSD 3-Clause License

RHC class: Randomized Hill Climbing optimizer for a PyTorch neural network model

Inspired by the PyTorch SGD implementation
https://github.com/pytorch/pytorch/blob/main/torch/optim/sgd.py
"""

from typing import Optional, Callable

import torch
import numpy as np


class RHC(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        step_size: float = 0.1,
        n_restarts: int = 1,
        max_no_improve: int = 1,
        random_state: int = 42,
    ):
        """
        Randomized Hill Climbing optimizer

        PARAMETERS:

        step_size {float}:
            Step size for hill climbing.

        n_restarts {int}:
            Number of restarts for the RHC.

        max_no_improve {int}:
            Number of iterations with no improvement before a restart.
        """
        random = np.random.default_rng(seed=random_state)
        torch.manual_seed(random_state)
        defaults = dict(
            step_size=step_size,
            n_restarts=n_restarts,
            max_no_improve=max_no_improve,
            random=random,
        )
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure: Callable[[], float]) -> Optional[float]:
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        for group in self.param_groups:
            params: list[torch.Tensor] = group["params"]

            loss = rhc(
                params,
                step_size=group["step_size"],
                n_restarts=group["n_restarts"],
                max_no_improve=group["max_no_improve"],
                random=group["random"],
                closure=closure,
            )
        return loss


def rhc(
    params: list[torch.Tensor],
    random: np.random.Generator,
    closure: Callable[[], float],
    step_size: float = 0.1,
    max_no_improve: int = 1,
    n_restarts: int = 1,
):
    r"""Functional API that performs Randomized Hill Climbing algorithm computation.

    See :class:`~pyperch.optim.RHC` for details.
    """
    loss = closure()
    no_improve_count = 0
    best_state = [param.clone() for param in params]

    restarts = n_restarts

    while restarts > 0:
        param_idx = random.integers(0, len(params))
        param = params[param_idx]
        old_param = param.clone()
        new_param = param.clone()

        flat_param = new_param.view(-1)
        idx = random.integers(0, len(flat_param))
        flat_param[idx] += step_size * random.choice([-1, 1])

        param.copy_(flat_param.view_as(param))
        new_loss = closure()

        if new_loss < loss:
            loss = new_loss
            no_improve_count = 0
            best_state[param_idx] = param.clone()
        else:
            param.copy_(old_param)
            no_improve_count += 1

        if no_improve_count >= max_no_improve:
            restarts -= 1
            for i, param in enumerate(params):
                params[i].copy_(best_state[i] + torch.randn_like(param) * step_size)
            no_improve_count = 0

    for i, param in enumerate(params):
        params[i].copy_(best_state[i])
    return loss
