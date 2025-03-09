"""
Author: Jakub Owczarek 
BSD 3-Clause License

SA class: Simulated Annealing optimizer for a PyTorch neural network model

Inspired by the PyTorch SGD implementation
https://github.com/pytorch/pytorch/blob/main/torch/optim/sgd.py
"""

import math
from typing import Optional, Callable

import torch
import numpy as np


class SA(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        t: float = 1,
        t_min: float = 0.1,
        step_size: float = 0.1,
        cooling: float = 0.95,
        random_state: int = 42,
    ):
        """
        Simulated Annealing optimizer

        PARAMETERS:

        t {float}:
            SA temperature.

        cooling {float}:
            Cooling rate.

        t_min {float}:
            SA minimum temperature.

        step_size {float}:
            Step size for hill climbing.

        random_state {int}:
            Random state for the optimizer.
        """
        random = np.random.default_rng(seed=random_state)
        torch.manual_seed(random_state)
        defaults = dict(
            t=t,
            t_min=t_min,
            step_size=step_size,
            cooling=cooling,
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

            loss = sa(
                params,
                t=group["t"],
                t_min=group["t_min"],
                step_size=group["step_size"],
                random=group["random"],
                cooling=group["cooling"],
                closure=closure,
            )
        return loss


def sa(
    params: list[torch.Tensor],
    random: np.random.Generator,
    closure: Callable[[], float],
    t: float = 1.0,
    t_min: float = 0.1,
    step_size: float = 0.1,
    cooling: float = 0.95,
):
    r"""Functional API that performs Simulated Annealing algorithm computation.

    See :class:`~pyperch.optim.SA` for details.
    """
    loss = closure()

    param = params[random.integers(0, len(params))]
    old_param = param.clone()

    flat_param = param.view(-1)
    idx = random.integers(0, len(flat_param))

    flat_param[idx] += step_size * random.choice([-1, 1])

    param.copy_(flat_param.view_as(param))
    new_loss = closure()

    delta = new_loss - loss
    accept = delta < 0 or random.random() < math.exp(-delta / t)

    if accept:
        loss = new_loss
    else:
        param.copy_(old_param)

    t = max(t * cooling, t_min)

    return loss
