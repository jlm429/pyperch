"""Simulated annealing optimizer.

pyperch.optim.sa.sa implementation placeholder
"""

from __future__ import annotations

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
) -> float:
    raise NotImplementedError("pyperch.optim.sa.sa implementation placeholder")
