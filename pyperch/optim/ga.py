"""Genetic Algorithm optimizer.

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


def ga(
    params: List[torch.Tensor],
    random: np.random.Generator,
    closure: Closure,
    population_size: int = 50,
    mutation_rate: float = 0.1,
    step_size: float = 0.1,
) -> float:
    """Functional Genetic Algorithm step.

    Mirrors the behavior of GA optimizer / ABAGAIL-style GA:

    - Build a population around the current weights
    - Evaluate fitness (lower loss = higher fitness)
    - Sample parents via fitness-proportional selection
    - Crossover to produce children
    - Elite selection for the rest
    - Mutate a portion of the population
    - Re-evaluate and adopt the best individual with accept/reject

    Only parameters with requires_grad=True are evolved.
    """
    # Filter to trainable params, like the Optimizer version
    trainable_params: List[Tensor] = [p for p in params if p.requires_grad]

    if not trainable_params:
        # Nothing to optimize
        return closure()

    # Baseline loss for accept/reject across all params
    old_loss = closure()

    with torch.no_grad():
        for param in trainable_params:
            # -----------------------------
            # 1. Initialize population
            # -----------------------------
            population = _initialize_population(population_size, param, step_size)

            # -----------------------------
            # 2. Evaluate population
            # -----------------------------
            probs, values = _evaluate(param, population, closure)

            # Derive GA hyperparams to-mate / to-mutate
            to_mate = max(1, population_size // 2)
            to_mutate = max(
                0, min(population_size, int(mutation_rate * population_size))
            )

            # -----------------------------
            # 3. Crossover
            # -----------------------------
            children, child_values = _crossover(population, probs, random, to_mate)

            # -----------------------------
            # 4. Selection (elitism)
            # -----------------------------
            next_pop, next_vals = _selection(
                population,
                children,
                values,
                child_values,
                probs,
                random,
                to_mate,
            )

            # -----------------------------
            # 5. Mutation
            # -----------------------------
            next_pop = _mutate(next_pop, step_size, random, to_mutate)

            # -----------------------------
            # 6. Re-evaluate updated pop
            # -----------------------------
            next_vals = _reevaluate(param, next_pop, next_vals, closure)

            # -----------------------------
            # 7. Update model weights
            # -----------------------------
            best_idx = int(np.argmin(next_vals))
            best_weights = next_pop[best_idx]

            old_param = param.clone()
            param.copy_(best_weights)

            new_loss = closure()
            if new_loss > old_loss:
                # Reject: revert to previous weights
                param.copy_(old_param)
            else:
                # Accept and update running best loss
                old_loss = new_loss

    return float(old_loss)


# --------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------


def _initialize_population(
    pop_size: int,
    param: Tensor,
    step_size: float,
) -> Tensor:
    """Initialize a population around current param using Gaussian noise."""
    return torch.stack(
        [param + torch.randn_like(param) * step_size for _ in range(pop_size)]
    )


def _evaluate(
    param: Tensor,
    population: Tensor,
    closure: Closure,
):
    """Evaluate each individual in the population and build selection probs.

    Mirrors the 'evaluate' helper from the working GA:

    - Temporarily copy individuals into the model param
    - Compute losses
    - Convert to minimization-based probabilities
    """
    old_param = param.clone()
    values: list[float] = []

    for indiv in population:
        param.copy_(indiv)
        loss = closure()
        values.append(float(loss))

    # Convert losses to probabilities (lower loss -> higher probability)
    arr = np.asarray(values, dtype=float)
    # scores proportional to "goodness": max - loss
    scores = arr.max() - arr + 1e-8

    if not np.isfinite(scores).any() or scores.sum() <= 0:
        probs = np.full_like(scores, 1.0 / len(scores), dtype=float)
    else:
        probs = scores / scores.sum()

    # Restore original parameter
    param.copy_(old_param)
    return probs, values


def _crossover(
    pop: Tensor,
    probs: np.ndarray,
    rng: np.random.Generator,
    to_mate: int,
):
    """ABAGAIL-style crossover: pick parents by roulette wheel and mix bits."""
    if to_mate <= 0:
        # No children
        return torch.empty((0, *pop.shape[1:]), device=pop.device, dtype=pop.dtype), []

    idx = rng.choice(len(pop), size=(to_mate, 2), p=probs)
    idx_t = torch.as_tensor(idx, dtype=torch.long, device=pop.device)

    p1 = pop[idx_t[:, 0]]
    p2 = pop[idx_t[:, 1]]

    # Uniform crossover mask
    mask = torch.bernoulli(torch.full_like(p1, 0.5))
    children = mask * p1 + (1 - mask) * p2
    # Values are unknown until evaluated
    child_values = [None] * to_mate
    return children, child_values


def _selection(
    pop: Tensor,
    children: Tensor,
    old_vals: list[float],
    child_vals: list[float],
    probs: np.ndarray,
    rng: np.random.Generator,
    to_mate: int,
):
    """Selection with elitism

    - First fill slots with children.
    - Remaining slots: sample survivors from old population.
    """
    pop_size = len(pop)
    survivors = []

    survivors_needed = pop_size - to_mate
    for _ in range(survivors_needed):
        idx = int(rng.choice(len(pop), p=probs))
        survivors.append(pop[idx].clone())

    if survivors:
        survivors_tensor = torch.stack(survivors, dim=0)
        next_pop = torch.cat([children, survivors_tensor], dim=0)
    else:
        next_pop = children

    # here we just carry old_vals
    # (they will be overwritten in _reevaluate anyway).
    combined_vals = np.array(
        list(old_vals[:to_mate]) + [v for v in old_vals[to_mate:]], dtype=float
    )
    return next_pop, combined_vals


def _mutate(
    pop: Tensor,
    strength: float,
    rng: np.random.Generator,
    to_mutate: int,
) -> Tensor:
    """Gaussian mutation of a subset of individuals."""
    pop_size = len(pop)
    if to_mutate <= 0 or pop_size == 0:
        return pop

    to_mutate = min(to_mutate, pop_size)
    idx = rng.choice(pop_size, size=to_mutate, replace=False)
    idx_t = torch.as_tensor(idx, dtype=torch.long, device=pop.device)

    noise = torch.randn_like(pop[idx_t]) * strength
    pop[idx_t] += noise
    return pop


def _reevaluate(
    param: Tensor,
    pop: Tensor,
    values,
    closure: Closure,
):
    """Re-evaluate the whole population after mutation/selection."""
    values = list(values)
    for i in range(len(pop)):
        param.copy_(pop[i])
        loss = closure()
        values[i] = float(loss)
    return values
