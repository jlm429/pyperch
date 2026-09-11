"""Genetic Algorithm optimizer.

Randomized Optimization methods for PyPerch.

Based on the original PyPerch optimizers by Jakub Owczarek
(BSD 3-Clause License).

These were also inspired by ABAGAIL's randomized optimization algorithms:
https://github.com/pushkar/ABAGAIL.

Substantial refactoring and redesign by John Mansfield (2026).
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from .base import RandomizedOptimizer


class GA(RandomizedOptimizer):
    """Persistent-population genetic algorithm for arbitrary PyTorch models.

    The optimizer initializes one population around the current parameters, then
    persists it across generations while applying elitist selection, crossover,
    and mutation. The model follows the best individual in the current population.

    Lower loss is assumed to be better.
    """

    _group_options = frozenset(
        {"initialization_step_size", "mutation_rate", "mutation_step_size"}
    )
    _optimizer_level_options = frozenset({"random_state", "population_size"})

    def __init__(
        self,
        params,
        population_size: int = 50,
        mutation_rate: float = 0.1,
        initialization_step_size: float = 0.1,
        mutation_step_size: float = 0.1,
        random_state: int | None = None,
    ):
        if population_size < 2:
            raise ValueError("population_size must be at least 2.")
        if mutation_rate < 0 or mutation_rate > 1:
            raise ValueError("mutation_rate must be in the interval [0, 1].")
        if initialization_step_size <= 0:
            raise ValueError("initialization_step_size must be positive.")
        if mutation_step_size <= 0:
            raise ValueError("mutation_step_size must be positive.")

        defaults = {
            "initialization_step_size": initialization_step_size,
            "mutation_rate": mutation_rate,
            "mutation_step_size": mutation_step_size,
        }
        super().__init__(params, defaults)

        self.population_size = population_size
        self._generator = self._make_generator(random_state)

    def reset_counters(self) -> None:
        """Reset counters and run-specific state without changing parameters."""
        super().reset_counters()
        self._initialized = False
        self._current_loss: float | None = None
        self._best_params: list[torch.Tensor] | None = None
        self._population: list[list[torch.Tensor]] | None = None
        self._population_losses: list[float] | None = None

    def step(self, closure: Callable[[], torch.Tensor]) -> torch.Tensor:
        """Initialize the population or evolve it by one generation."""
        if closure is None:
            raise ValueError("GA requires a closure that returns the loss.")

        if not self._initialized or self._population is None:
            return self._initialize(closure)

        assert self._population_losses is not None
        assert self._current_loss is not None
        previous_loss = self._current_loss

        elites, elite_losses = self._select_population(
            self._population,
            self._population_losses,
        )
        children = self._mutate(self._crossover(elites))
        next_population = elites + children
        next_losses: list[float | None] = elite_losses + [None] * len(children)
        result_template = self._evaluate_unknown(
            next_population,
            next_losses,
            closure,
        )
        known_losses = [float(value) for value in next_losses]
        best_idx = self._best_index(known_losses)
        current_loss = known_losses[best_idx]

        self._population = next_population
        self._population_losses = known_losses
        self._restore_params(next_population[best_idx])
        self._current_loss = current_loss
        self.proposed_steps += 1

        if current_loss < previous_loss:
            self.accepted_steps += 1
            if self.best_loss is None or current_loss < self.best_loss:
                self.best_loss = current_loss
                self._best_params = self._clone_all_params()
        else:
            self.rejected_steps += 1

        return result_template.detach().new_tensor(current_loss)

    def _initialize(self, closure: Callable[[], torch.Tensor]) -> torch.Tensor:
        loss_tensor = self._evaluate_loss(closure)
        base_loss = float(loss_tensor.detach().item())
        population = self._initialize_population(self._clone_params())
        losses: list[float | None] = [base_loss] + [None] * (self.population_size - 1)
        result_template = self._evaluate_unknown(
            population,
            losses,
            closure,
            result_template=loss_tensor,
        )
        known_losses = [float(value) for value in losses]
        best_idx = self._best_index(known_losses)
        current_loss = known_losses[best_idx]

        self._restore_params(population[best_idx])
        self._initialized = True
        self._population = population
        self._population_losses = known_losses
        self._current_loss = current_loss
        self.best_loss = current_loss
        self._best_params = self._clone_all_params()

        return result_template.detach().new_tensor(current_loss)

    def _evaluate_loss(
        self,
        closure: Callable[[], torch.Tensor],
    ) -> torch.Tensor:
        with torch.enable_grad():
            loss_tensor = closure()

        self._record_eval()
        return loss_tensor

    @torch.no_grad()
    def _initialize_population(
        self,
        base_params: list[torch.Tensor],
    ) -> list[list[torch.Tensor]]:
        population = [base_params]

        for _ in range(self.population_size - 1):
            individual = []
            for param, (_, group) in zip(
                base_params,
                self._parameters_with_groups(),
            ):
                noise = self._randn_like(param)
                individual.append(param + group["initialization_step_size"] * noise)
            population.append(individual)

        return population

    def _evaluate_unknown(
        self,
        population: list[list[torch.Tensor]],
        losses: list[float | None],
        closure: Callable[[], torch.Tensor],
        result_template: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for index, individual in enumerate(population):
            if losses[index] is not None:
                continue
            self._restore_params(individual)
            with torch.enable_grad():
                result_template = closure()
            losses[index] = float(result_template.detach().item())
            self._record_eval()

        assert result_template is not None
        return result_template

    def _select_population(
        self,
        population: list[list[torch.Tensor]],
        losses: list[float],
    ) -> tuple[list[list[torch.Tensor]], list[float]]:
        keep_count = max(1, self.population_size // 2)
        ranked_indices = sorted(
            range(len(losses)),
            key=lambda index: (losses[index], index),
        )
        elite_indices = ranked_indices[:keep_count]

        return (
            [self._clone_individual(population[index]) for index in elite_indices],
            [losses[index] for index in elite_indices],
        )

    @torch.no_grad()
    def _crossover(
        self,
        parents: list[list[torch.Tensor]],
    ) -> list[list[torch.Tensor]]:
        children = []
        for _ in range(self.population_size - len(parents)):
            idx1 = torch.randint(
                low=0,
                high=len(parents),
                size=(1,),
                generator=self._generator,
            ).item()
            idx2 = torch.randint(
                low=0,
                high=len(parents),
                size=(1,),
                generator=self._generator,
            ).item()
            child = []
            for param1, param2 in zip(parents[idx1], parents[idx2]):
                mask = self._rand_like(param1) < 0.5
                child.append(torch.where(mask, param1, param2))
            children.append(child)

        return children

    @torch.no_grad()
    def _mutate(
        self,
        population: list[list[torch.Tensor]],
    ) -> list[list[torch.Tensor]]:
        mutated = []
        parameter_groups = self._parameters_with_groups()

        for individual in population:
            new_individual = []
            for param, (_, group) in zip(individual, parameter_groups):
                mutation_mask = self._rand_like(param) < group["mutation_rate"]
                noise = self._randn_like(param)
                new_individual.append(
                    param + mutation_mask * group["mutation_step_size"] * noise
                )
            mutated.append(new_individual)

        return mutated

    @staticmethod
    def _clone_individual(
        individual: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        return [param.detach().clone() for param in individual]

    @staticmethod
    def _best_index(losses: list[float]) -> int:
        return min(range(len(losses)), key=lambda index: (losses[index], index))

    @torch.no_grad()
    def restore_best(self) -> None:
        """Restore the best parameters observed so far."""
        if self._best_params is not None:
            self._restore_all_params(self._best_params)
            self._current_loss = self.best_loss
            self._initialized = True

    def _validate_group_options(self, param_group) -> None:
        if "mutation_rate" in param_group:
            mutation_rate = param_group["mutation_rate"]
            if mutation_rate < 0 or mutation_rate > 1:
                raise ValueError(
                    "mutation_rate must be in [0, 1] in every parameter group."
                )
        for name in ("initialization_step_size", "mutation_step_size"):
            if name in param_group and param_group[name] <= 0:
                raise ValueError(f"{name} must be positive in every parameter group.")

    def _algorithm_checkpoint_state(self) -> dict:
        population = None
        if self._population is not None:
            population = [
                self._clone_individual(individual) for individual in self._population
            ]
        return {
            "population_size": self.population_size,
            "population": population,
            "population_losses": (
                None
                if self._population_losses is None
                else list(self._population_losses)
            ),
        }

    def _load_algorithm_checkpoint_state(self, state: dict) -> None:
        population_size = state["population_size"]
        if population_size < 2:
            raise ValueError("Checkpoint population_size must be at least 2.")
        self.population_size = population_size

        population = state.get("population")
        losses = state.get("population_losses")
        if population is None or losses is None:
            if population is not None or losses is not None:
                raise ValueError(
                    "Checkpoint GA population and population_losses must both be set."
                )
            self._population = None
            self._population_losses = None
            return
        if len(population) != population_size or len(losses) != population_size:
            raise ValueError("Checkpoint GA population does not match population_size.")

        parameters = self._parameters()
        restored_population = []
        for individual in population:
            if len(individual) != len(parameters):
                raise ValueError(
                    "Checkpoint GA individual does not match the parameter count."
                )
            restored_individual = []
            for parameter, value in zip(parameters, individual):
                if value.shape != parameter.shape:
                    raise ValueError(
                        "Checkpoint GA individual does not match parameter shapes."
                    )
                restored_individual.append(
                    value.detach()
                    .to(device=parameter.device, dtype=parameter.dtype)
                    .clone()
                )
            restored_population.append(restored_individual)

        self._population = restored_population
        self._population_losses = [float(loss) for loss in losses]

    def _migrate_legacy_param_groups(self, param_groups: list[dict]) -> dict | None:
        population_sizes = [
            group.pop("population_size")
            for group in param_groups
            if "population_size" in group
        ]
        if not population_sizes:
            return None
        if (
            len(population_sizes) != len(param_groups)
            or len(set(population_sizes)) != 1
        ):
            raise ValueError(
                "Legacy GA state has conflicting per-group population_size values "
                "and cannot be migrated."
            )

        population_size = population_sizes[0]
        if population_size < 2:
            raise ValueError("Legacy GA population_size must be at least 2.")
        return {"population_size": population_size}

    def _load_legacy_algorithm_state(self, state: dict | None) -> None:
        if state is not None:
            self.population_size = state["population_size"]
