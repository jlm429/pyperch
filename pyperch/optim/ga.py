""""
Author: Jakub Owczarek 
BSD 3-Clause License

GA class: Genetic Algorithm optimizer for a PyTorch neural network model

"""


import torch
import numpy as np
from torch import Tensor
from typing import Callable, Optional, List


class GA(torch.optim.Optimizer):
    """
    Standalone Genetic Algorithm with full freezing support.
    Only parameters with requires_grad=True participate in evolution.
    """

    def __init__(
        self,
        params,
        mutation_strength: float = 0.1,
        random_state: int = 42,
        population_size: int = 300,
        to_mate: int = 150,
        to_mutate: int = 50,
    ):
        rng = np.random.default_rng(seed=random_state)
        torch.manual_seed(random_state)

        params = list(params)

        # --- identify trainable parameters ---
        trainable_indices = [i for i, p in enumerate(params) if p.requires_grad]
        trainable_params = [params[i] for i in trainable_indices]

        # --- initialize populations only for trainable params ---
        populations = initialize_populations(population_size, trainable_params)

        defaults = dict(
            mutation_strength=mutation_strength,
            random=rng,
            population_size=population_size,
            to_mate=to_mate,
            to_mutate=to_mutate,
            populations=populations,
            trainable_indices=trainable_indices,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], float]) -> Optional[float]:
        loss = None

        for group in self.param_groups:
            params = group["params"]
            train_ids = group["trainable_indices"]

            # trainable-only view
            trainable_params = [params[i] for i in train_ids]

            loss, new_pops = ga_step(
                params=trainable_params,
                populations=group["populations"],
                rng=group["random"],
                mutation_strength=group["mutation_strength"],
                population_size=group["population_size"],
                to_mate=group["to_mate"],
                to_mutate=group["to_mutate"],
                closure=closure,
            )

            group["populations"] = new_pops

        return loss


def ga_step(
    params: List[Tensor],
    populations: List[Tensor],
    rng,
    mutation_strength: float,
    population_size: int,
    to_mate: int,
    to_mutate: int,
    closure: Callable[[], float],
):
    """
    GA - inspired by https://github.com/pushkar/ABAGAIL:

    - Evaluate population fitness
    - Crossover top performers
    - Mutate some portion
    - Select survivors
    - Replace actual parameters with best individual
    """

    old_loss = closure()
    new_pops = []

    for param, population in zip(params, populations):

        # === Evaluate population ===
        probs, values = evaluate(param, population, closure)

        # === Crossover ===
        children, child_values = crossover(population, probs, rng, to_mate)

        # === Selection ===
        next_pop, next_vals = selection(
            population, children, values, child_values, probs, rng, to_mate
        )

        # === Mutation ===
        next_pop = mutate(next_pop, mutation_strength, rng, to_mutate)

        # === Re-evaluate ===
        next_vals = reevaluate(param, next_pop, next_vals, closure)

        # === Replace model weights ===
        best_idx = np.argmin(next_vals)
        best_weights = next_pop[best_idx]
        old_param = param.clone()
        param.copy_(best_weights)

        new_loss = closure()
        if new_loss > old_loss:  # reject
            param.copy_(old_param)
        else:
            old_loss = new_loss  # accept

        new_pops.append(next_pop.clone())

    return old_loss, new_pops


# -------------------------
# Utility functions 
# -------------------------

def initialize_populations(pop_size, params):
    return [
        torch.stack([
            p + torch.randn_like(p) * 0.1
            for _ in range(pop_size)
        ])
        for p in params
    ]


def evaluate(param, population, closure):
    old_param = param.clone()
    values = []

    for indiv in population:
        param.copy_(indiv)
        loss = closure()
        values.append(loss)

    # convert to minimization-probabilities
    arr = np.array(values)
    scores = arr.max() - arr + 1e-8
    probs = scores / scores.sum()

    param.copy_(old_param)
    return probs, values


def crossover(pop, probs, rng, to_mate):
    idx = rng.choice(len(pop), (to_mate, 2), p=probs)
    idx = torch.tensor(idx, dtype=torch.long, device=pop.device)

    p1 = pop[idx[:, 0]]
    p2 = pop[idx[:, 1]]

    mask = torch.bernoulli(torch.full_like(p1, 0.5))
    children = mask * p1 + (1 - mask) * p2
    return children, [None] * to_mate


def selection(pop, children, old_vals, child_vals, probs, rng, to_mate):
    survivors = []

    for _ in range(len(pop) - to_mate):
        idx = rng.choice(len(pop), p=probs)
        survivors.append(pop[idx].clone())

    combined = torch.cat([children, torch.stack(survivors)], dim=0)
    combined_vals = np.array(
        list(old_vals[:to_mate]) + [v for v in old_vals[to_mate:]]
    )

    return combined, combined_vals


def mutate(pop, strength, rng, to_mutate):
    if to_mutate == 0:
        return pop

    idx = rng.choice(len(pop), to_mutate, replace=False)
    noise = torch.randn_like(pop[idx]) * strength
    pop[idx] += noise
    return pop


def reevaluate(param, pop, values, closure):
    for i in range(len(pop)):
        param.copy_(pop[i])
        values[i] = closure()
    return values

