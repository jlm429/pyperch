"""
Author: Jakub Owczarek 
BSD 3-Clause License

SA class: Simulated Annealing optimizer for a PyTorch neural network model

"""
import math
from typing import Callable, Optional, List

import numpy as np
import torch
from torch import Tensor


class SA(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        t: float = 1.0,
        t_min: float = 0.1,
        cooling: float = 0.95,
        step_size: float = 0.1,
        random_state: int = 42,
    ):
        rng = np.random.default_rng(seed=random_state)
        torch.manual_seed(random_state)

        defaults = dict(
            t=t,
            t_min=t_min,
            cooling=cooling,
            step_size=step_size,
            random=rng,
        )

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], float]) -> Optional[float]:
        """
        Perform a single SA update over all parameter groups.
        Must update group["t"] each call.
        """
        loss = None

        for group in self.param_groups:
            params: List[Tensor] = group["params"]

            loss, new_t = sa_step(
                params=params,
                closure=closure,
                t=group["t"],
                t_min=group["t_min"],
                cooling=group["cooling"],
                step_size=group["step_size"],
                rng=group["random"],
            )

            group["t"] = new_t

        return loss


def sa_step(
    params: List[Tensor],
    closure: Callable[[], float],
    t: float,
    t_min: float,
    cooling: float,
    step_size: float,
    rng: np.random.Generator,
):
    """
    Perform ONE simulated annealing step.

    Mirrors skorch SAModule.run_sa_single_step - inspired by https://github.com/pushkar/ABAGAIL:
      - Select ONLY trainable parameters (requires_grad=True)
      - Choose ONE parameter tensor
      - Choose ONE weight within that tensor
      - Add ±step_size
      - Evaluate closure() before/after
      - Accept/reject based on SA probability
      - Update temperature and return it
    """

    # ---- 1. Compute current loss ----
    old_loss = closure()

    # ---- 2. Pick only trainable parameters ----
    trainable_params = [p for p in params if p.requires_grad]
    if not trainable_params:
        # no trainable params → do nothing
        return old_loss, t

    # ---- 3. Select ONE parameter tensor ----
    p = trainable_params[rng.integers(0, len(trainable_params))]

    # Save old version
    old_param = p.clone()

    # ---- 4. Mutate ONE element inside this parameter ----
    flat = p.view(-1)
    idx = rng.integers(0, flat.numel())
    change = step_size * rng.choice([-1, 1])
    flat[idx] += change
    p.copy_(flat.view_as(p))

    # ---- 5. Compute new loss ----
    new_loss = closure()
    delta = new_loss - old_loss

    # ---- 6. Decide acceptance ----
    if delta > 0:
        prob = math.exp(-delta / t)
        if rng.random() >= prob:
            # reject → revert weights
            p.copy_(old_param)
            new_loss = old_loss

    # ---- 7. Cool temperature ----
    new_t = max(t * cooling, t_min)

    return new_loss, new_t
