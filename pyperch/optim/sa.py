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
    """
    Standalone Simulated Annealing optimizer for PyTorch.
    """

    def __init__(
        self,
        params,
        t: float = 1.0,
        t_min: float = 0.1,
        cooling: float = 0.95,
        step_size: float = 0.1,
        random_state: int = 42,
    ):
        self.rng = np.random.default_rng(seed=random_state)
        torch.manual_seed(random_state)

        defaults = dict(
            t=t,
            t_min=t_min,
            cooling=cooling,
            step_size=step_size,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], float]) -> Optional[float]:
        """
        SA step - inspired by https://github.com/pushkar/ABAGAIL:
        - Compute loss
        - Pick ONE trainable parameter tensor
        - Apply full-vector perturbation
        - Accept/reject based on SA probability
        - Cool temperature
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
                rng=self.rng,
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
    SA:
      - Select one trainable tensor
      - Create full neighbor (vector noise)
      - Accept w/ SA prob
      - Cool temperature
    """

    # ---- 1. Current loss ----
    old_loss = closure()

    # ---- 2. Get trainable tensors ----
    trainable = [p for p in params if p.requires_grad]
    if not trainable:
        return old_loss, t

    # ---- 3. Pick ONE tensor ----
    p = trainable[rng.integers(0, len(trainable))]
    old_p = p.clone()

    # ---- 4. Full neighbor perturbation ----
    noise = (rng.random(p.numel()) - 0.5).reshape(p.shape)
    noise = torch.tensor(noise, dtype=p.dtype, device=p.device)

    p[:] = p + step_size * noise

    # ---- 5. Evaluate mutated model ----
    new_loss = closure()
    delta = new_loss - old_loss

    # ---- 6. SA acceptance rule ----
    if delta > 0:
        prob = math.exp(-delta / max(t, 1e-9))
        if rng.random() >= prob:
            # reject
            p.copy_(old_p)
            new_loss = old_loss

    # ---- 7. Cool temperature ----
    new_t = max(t * cooling, t_min)

    return new_loss, new_t

