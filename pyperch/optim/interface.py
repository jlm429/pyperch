from __future__ import annotations

from typing import Callable, List

import numpy as np
import torch

from ..config.schema import OptimizerConfig


Closure = Callable[[], float]


def get_optimizer(name: str):
    name = name.lower()
    if name == "sa":
        from . import sa

        return sa.sa
    if name == "rhc":
        from . import rhc

        return rhc.rhc
    if name == "ga":
        from . import ga

        return ga.ga
    raise KeyError(f"Unknown optimizer: {name}")


def run_optimizer_step(
    name: str,
    params: List[torch.Tensor],
    rng: np.random.Generator,
    closure: Closure,
    cfg: OptimizerConfig,
) -> float:
    name = name.lower()
    opt = get_optimizer(name)

    if name == "sa":
        loss, new_t = opt(
            params=params,
            random=rng,
            closure=closure,
            t=cfg.t,
            t_min=cfg.t_min,
            step_size=cfg.step_size,
            cooling=cfg.cooling,
        )
        cfg.t = new_t  # persist temperature like group["t"] in the old optimizer
        return float(loss)

    elif name == "rhc":
        loss = opt(
            params=params,
            random=rng,
            closure=closure,
            step_size=cfg.step_size,
        )
        return float(loss)

    elif name == "ga":
        loss = opt(
            params=params,
            random=rng,
            closure=closure,
            population_size=cfg.population_size,
            mutation_rate=cfg.mutation_rate,
            step_size=cfg.step_size,
        )
        return float(loss)

    else:
        raise KeyError(f"Unknown optimizer: {name}")
