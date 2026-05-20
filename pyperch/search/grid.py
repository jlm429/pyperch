from __future__ import annotations

from collections.abc import Callable
from itertools import product


def grid_search(
    param_grid: dict, objective: Callable[[dict], float]
) -> tuple[dict, float]:
    """Evaluate every parameter combination and return the best result."""
    keys = list(param_grid.keys())
    best_params = None
    best_score = None

    for values in product(*(param_grid[key] for key in keys)):
        params = dict(zip(keys, values))
        score = objective(params)
        if best_score is None or score < best_score:
            best_params = params
            best_score = score

    return best_params, best_score
