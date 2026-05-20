from __future__ import annotations

import random
from collections.abc import Callable


def random_search(
    param_space: dict,
    objective: Callable[[dict], float],
    n_iter: int = 10,
    seed: int | None = None,
) -> tuple[dict, float]:
    """Sample parameter combinations and return the best result."""
    rng = random.Random(seed)
    best_params = None
    best_score = None

    for _ in range(n_iter):
        params = {key: rng.choice(values) for key, values in param_space.items()}
        score = objective(params)
        if best_score is None or score < best_score:
            best_params = params
            best_score = score

    return best_params, best_score
