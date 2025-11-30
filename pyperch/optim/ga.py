""""
Author: Jakub Owczarek 
BSD 3-Clause License

GA class: Genetic Algorithm optimizer for a PyTorch neural network model

Inspired by the PyTorch SGD implementation
https://github.com/pytorch/pytorch/blob/main/torch/optim/sgd.py
"""

from typing import Optional, Callable

import torch
import numpy as np


class GA(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        mutation_strength: float = 0.1,
        random_state: int = 42,
        population_size: int = 300,
        to_mate: int = 150,
        to_mutate: int = 50,
    ):
        """
        Genetic Algorithm optimizer

        PARAMETERS:

        population_size {int}:
            GA population size. Must be greater than 0.

        to_mate {int}:
            GA size of population to mate each time step.

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
            to_mate=to_mate,
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

            loss = ga(
                params,
                populations=group["populations"],
                mutation_strength=group["mutation_strength"],
                random=group["random"],
                population_size=group["population_size"],
                to_mate=group["to_mate"],
                to_mutate=group["to_mutate"],
                closure=closure,
            )

        return loss


def ga(
    params,
    closure: Callable[[], float],
    random,
    populations,
    mutation_strength: float = 0.1,
    population_size: int = 300,
    to_mate: int = 150,
    to_mutate: int = 50,
):
    r"""Functional API that performs Genetic Algorithm algorithm computation.

    See :class:`~pyperch.optim.GA` for details.
    """
    loss = closure()

    for param, population in zip(params, populations):
        old_param = param.clone()

        probabilities, values = evaluate(
            param=param,
            population=population,
            closure=closure,
            population_size=population_size,
        )

        new_population, new_values = crossover(
            population=population,
            population_size=population_size,
            to_mate=to_mate,
            random=random,
            probabilities=probabilities,
        )

        population, values = selection(
            population=population,
            new_population=new_population,
            probabilities=probabilities,
            random=random,
            to_mate=to_mate,
            population_size=population_size,
            values=values,
            new_values=new_values,
        )

        population = mutate(
            population=population,
            population_size=population_size,
            to_mutate=to_mutate,
            random=random,
            mutation_strength=mutation_strength,
        )

        values = reevaluate(
            param=param,
            population=population,
            population_size=population_size,
            values=values,
            closure=closure,
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


def initialize_populations(
    population_size: int, params: list[torch.Tensor]
) -> list[torch.Tensor]:
    return [
        torch.stack(
            [param + torch.randn_like(param) * 0.1 for _ in range(population_size)]
        )
        for param in params
    ]


def evaluate(
    param: torch.Tensor,
    population: torch.Tensor,
    closure: Callable[[], float],
    population_size: int,
) -> tuple[np.ndarray, list[float]]:
    old_param = param.clone()

    values = []
    for new_param in population:
        param.copy_(new_param)
        loss = closure()
        values.append(-loss)

    probabilities = np.array(values)
    probabilities -= probabilities.min()

    if probabilities.sum() > 0:
        probabilities /= probabilities.sum()
    else:
        probabilities = np.ones(population_size) / population_size

    param.copy_(old_param)

    return probabilities, values


def crossover(
    population: torch.Tensor,
    population_size: int,
    to_mate: int,
    random: np.random.Generator,
    probabilities: np.ndarray,
) -> tuple[list[torch.Tensor], np.ndarray]:
    parent_idxs = random.choice(population_size, (to_mate, 2), p=probabilities)
    parent_idxs = torch.tensor(parent_idxs, dtype=torch.int, device=population.device)
    parent1 = population[parent_idxs[:, 0]]
    parent2 = population[parent_idxs[:, 1]]

    mask = torch.bernoulli(torch.full_like(parent1, 0.5))

    children = mask * parent1 + (1 - mask) * parent2
    return list(children), np.zeros(population_size)


def selection(
    population: torch.Tensor,
    new_population: list[torch.Tensor],
    probabilities: np.ndarray,
    random: np.random.Generator,
    to_mate: int,
    population_size: int,
    values: list[float],
    new_values: np.ndarray,
) -> tuple[torch.Tensor, np.ndarray]:
    new_population_list = []
    for i in range(to_mate, population_size):
        index = random.choice(population_size, p=probabilities)
        new_population_list.append(population[index].clone())
        new_values[i] = values[i]

    combined_list = list(new_population) + new_population_list

    new_population_tensor = torch.stack(combined_list)

    return new_population_tensor, new_values


def mutate(
    population: torch.Tensor,
    population_size: int,
    to_mutate: int,
    random: np.random.Generator,
    mutation_strength: float,
) -> torch.Tensor:
    if to_mutate == 0:
        return
    new_population = population.clone()

    mutation_indices = random.choice(population_size, to_mutate, replace=False)

    sign = torch.tensor(
        random.choice([-1, 1]), dtype=torch.float, device=new_population[0].device
    )
    noise = (
        sign * torch.randn_like(new_population[mutation_indices]) * mutation_strength
    )

    new_population[mutation_indices] += noise

    return new_population


def reevaluate(
    param: torch.Tensor,
    population: torch.Tensor,
    population_size: int,
    values: np.ndarray,
    closure: Callable[[], float],
) -> np.ndarray:
    for i in range(population_size):
        if values[i] == -1:
            i_tensor = torch.tensor(i, dtype=torch.int, device=population.device)
            new_param = population[i_tensor]
            param.copy_(new_param)
            values[i] = closure()

    return values
