"""
Randomized Optimization methods for PyPerch.

Based on the original PyPerch optimizers by Jakub Owczarek
(BSD 3-Clause License).

These were also inspired by ABAGAIL’s randomized optimization algorithms - https://github.com/pushkar/ABAGAIL.

Substantial refactoring and redesign by John Mansfield (2025).

"""

from __future__ import annotations
from typing import Callable, List
import numpy as np
import torch
from torch import Tensor

Closure = Callable[[], float]


def rhc(
    params: List[Tensor],
    random: np.random.Generator,
    closure: Closure,
    step_size: float = 0.1,
) -> float:
    """
    Functional Randomized Hill Climbing step:

    - evaluate old loss
    - apply full-vector noise
    - evaluate new loss
    - accept if better, otherwise revert
    """
    # ---- 1. Compute current loss ----
    old_loss = closure()

    # ---- 2. Save backups of all parameters ----
    backups = [p.clone() for p in params]

    # ---- 3. Perturb parameters (no_grad required!) ----
    with torch.no_grad():
        for p in params:
            if not p.requires_grad:
                continue

            # ABAGAIL-style noise in [-0.5, 0.5]
            noise_np = (random.random(p.numel()) - 0.5).reshape(p.shape)
            noise = torch.tensor(noise_np, dtype=p.dtype, device=p.device)

            p.add_(step_size * noise)  # safe in-place update under no_grad

    # ---- 4. Compute new loss ----
    new_loss = closure()

    # ---- 5. Accept or revert ----
    if new_loss > old_loss:
        with torch.no_grad():
            for p, b in zip(params, backups):
                p.copy_(b)
        return old_loss

    return new_loss
