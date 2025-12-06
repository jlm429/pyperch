"""Simulated annealing optimizer.

Randomized Optimization methods for PyPerch.

Based on the original PyPerch optimizers by Jakub Owczarek
(BSD 3-Clause License).

These were also inspired by ABAGAIL’s randomized optimization algorithms - https://github.com/pushkar/ABAGAIL.

Substantial refactoring and redesign by John Mansfield (2025).
"""

from __future__ import annotations
from __future__ import annotations
import math
from typing import Callable, List

import numpy as np
import torch

Closure = Callable[[], float]


def sa(
    params: List[torch.Tensor],
    random: np.random.Generator,
    closure: Closure,
    t: float = 1.0,
    t_min: float = 0.1,
    step_size: float = 0.1,
    cooling: float = 0.95,
):
    """Stateless Simulated Annealing step.

    This mirrors the old SA + sa_step behavior as closely as possible:

    - Entire step runs under torch.no_grad() (like @torch.no_grad() on step()).
    - Select ONE trainable parameter tensor at random.
    - Apply full-vector perturbation with uniform noise in [-0.5, 0.5].
    - Accept/reject based on standard SA rule.
    - Cool the temperature once per call.

    Returns:
        (new_loss, new_t)
    """
    with torch.no_grad():
        # ---- 1. Current loss ----
        old_loss = closure()

        # ---- 2. Get trainable tensors ----
        trainable = [p for p in params if p.requires_grad]
        if not trainable:
            return old_loss, t

        # ---- 3. Pick ONE tensor ----
        p = trainable[random.integers(0, len(trainable))]
        old_p = p.clone()

        # ---- 4. Full neighbor perturbation ----
        noise_np = (random.random(p.numel()) - 0.5).reshape(p.shape)
        noise = torch.tensor(noise_np, dtype=p.dtype, device=p.device)

        # same style as old: p[:] = p + step_size * noise
        p[:] = p + step_size * noise

        # ---- 5. Evaluate mutated model ----
        new_loss = closure()
        delta = new_loss - old_loss

        # ---- 6. SA acceptance rule ----
        if delta > 0:
            prob = math.exp(-delta / max(t, 1e-9))
            if random.random() >= prob:
                # reject - restore old params
                p.copy_(old_p)
                new_loss = old_loss

        # ---- 7. Cool temperature ----
        new_t = max(t * cooling, t_min)

        return new_loss, new_t
