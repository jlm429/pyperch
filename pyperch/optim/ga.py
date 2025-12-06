"""Genetic algorithm optimizer.

pyperch.optim.ga.ga implementation placeholder
"""

from __future__ import annotations

from typing import Callable, List

import numpy as np
import torch


Closure = Callable[[], float]


def ga(
    params: List[torch.Tensor],
    random: np.random.Generator,
    closure: Closure,
    population_size: int = 50,
    mutation_rate: float = 0.1,
    step_size: float = 0.1,
) -> float:
    raise NotImplementedError("pyperch.optim.ga.ga implementation placeholder")
