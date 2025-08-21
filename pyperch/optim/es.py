"""
Author: Jakub Owczarek 
BSD 3-Clause License

ES class: Evolutionary Strategy optimizer for a PyTorch neural network model

Inspired by the PyTorch SGD implementation
https://github.com/pytorch/pytorch/blob/main/torch/optim/sgd.py
"""

from typing import Optional, Callable

import torch
import numpy as np

from pyperch.optim.ga import initialize_populations, mutate, evaluate, selection


class ES(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        mutation_strength: float = 1.0,
        random_state: int = 42,
        population_size: int = 100,
        to_mutate: int = 50,
    ):
        """
        Evolutionary Strategy optimizer

        PARAMETERS:

        population_size {int}:
            GA population size. Must be greater than 0.

        to_mutate {int}:
            GA size of population to mutate each time step.

        random_state {int}:
            Random state for the optimizer.

        mutation_strength {float}:
            Mutation strength.
        """
        random = np.random.default_rng(seed=random_state)
        torch.manual_seed(random_state)

        params = list(params)
        populations = initialize_populations(population_size, params)
        defaults = dict(
            mutation_strength=mutation_strength,
            random=random,
            population_size=population_size,
            to_mutate=to_mutate,
            populations=populations,
        )
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure: Callable[[], float]) -> Optional[float]:
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        for group in self.param_groups:
            params: list[torch.Tensor] = group["params"]

            loss = es(
                params,
                populations=group["populations"],
                mutation_strength=group["mutation_strength"],
                random=group["random"],
                population_size=group["population_size"],
                to_mutate=group["to_mutate"],
                closure=closure,
            )

        return loss


def es(
    params,
    closure: Callable[[], float],
    random,
    populations,
    mutation_strength: float = 1.0,
    population_size: int = 100,
    to_mutate: int = 50,
):
    r"""Functional API that performs Evolutionary Strategy algorithm computation.

    See :class:`~pyperch.optim.ES` for details.
    """

    loss = closure()

    for param, population in zip(params, populations):
        old_param = param.clone()

        new_population = mutate(
            population=population,
            population_size=population_size,
            to_mutate=to_mutate,
            random=random,
            mutation_strength=mutation_strength,
        )

        probabilities, values = evaluate(
            param=param,
            population=population,
            closure=closure,
            population_size=population_size,
        )

        new_values = [0] * len(values)
        population, values = selection(
            population=population,
            new_population=new_population,
            probabilities=probabilities,
            random=random,
            to_mate=0,
            population_size=population_size,
            values=values,
            new_values=new_values,
        )

        best_fitness_index = np.argmin(values)
        param.copy_(population[best_fitness_index])

        new_loss = closure()

        accept = new_loss < loss
        if accept:
            loss = new_loss
        else:
            param.copy_(old_param)

        return loss
