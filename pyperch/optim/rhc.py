"""
Author: Jakub Owczarek 
BSD 3-Clause License

RHC class: Randomized Hill Climbing optimizer for a PyTorch neural network model


"""

from typing import Callable, Optional, List
import numpy as np
import torch
from torch import Tensor


class RHC(torch.optim.Optimizer):
    """
    Standalone Randomized Hill Climbing optimizer for PyTorch - inspired by https://github.com/pushkar/ABAGAIL
    with freezing support (requires grad=False tensors)
    """

    def __init__(self, params, step_size: float = 0.1, random_state: int = 42):
        self.rng = np.random.default_rng(seed=random_state)
        torch.manual_seed(random_state)

        defaults = dict(step_size=step_size)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], float]) -> Optional[float]:
        """
        Hill-climbing step:
        - Compute current loss
        - Generate a full neighbor (all weights change) for trainable params
        - Accept if loss improves, else revert (greedy)
        """

        loss = None

        for group in self.param_groups:
            params: List[Tensor] = group["params"]
            step_size = group["step_size"]

            loss = rhc_step(
                params=params,
                closure=closure,
                step_size=step_size,
                rng=self.rng,
            )

        return loss


def rhc_step(
    params: List[Tensor],
    closure: Callable[[], float],
    step_size: float,
    rng: np.random.Generator,
):
    """
    1. full-vector noise for all params 
    2. revert if new loss worse 
    """

    # ---- 1. Compute current loss ----
    old_loss = closure()

    # Backup all trainable params
    backups = [p.clone() for p in params]

    # ---- 2. full neighbor perturbation ----
    for i, p in enumerate(params):
        if not p.requires_grad:
            continue

        # ABAGAIL: full random vector in [-0.5, 0.5]
        noise = (rng.random(p.numel()) - 0.5).reshape(p.shape)
        noise = torch.tensor(noise, dtype=p.dtype, device=p.device)

        p[:] = p + step_size * noise

    # ---- 3. Evaluate neighbor ----
    new_loss = closure()

    # ---- 4. Accept if better, else revert ----
    if new_loss > old_loss:
        # revert everything
        for p, b in zip(params, backups):
            p.copy_(b)
        return old_loss

    return new_loss
