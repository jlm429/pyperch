"""Genetic Algorithm optimizer.

Randomized Optimization methods for PyPerch.

Based on the original PyPerch optimizers by Jakub Owczarek
(BSD 3-Clause License).

These were also inspired by ABAGAIL’s randomized optimization algorithms - https://github.com/pushkar/ABAGAIL.

Substantial refactoring and redesign by John Mansfield (2026).
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from .base import RandomizedOptimizer


class GA(RandomizedOptimizer):
    """Genetic algorithm optimizer for arbitrary PyTorch models.

    This optimizer builds a population around the current parameters,
    evaluates candidate solutions, applies selection, crossover, mutation,
    and adopts the best candidate when it improves the loss.

    Lower loss is assumed to be better.
    """

    _group_options = frozenset({"mutation_rate", "step_size"})
    _optimizer_level_options = frozenset({"random_state", "population_size"})

    def __init__(
        self,
        params,
        population_size: int = 50,
        mutation_rate: float = 0.1,
        step_size: float = 0.1,
        random_state: int | None = None,
    ):
        if population_size < 2:
            raise ValueError("population_size must be at least 2.")
        if mutation_rate < 0 or mutation_rate > 1:
            raise ValueError("mutation_rate must be in the interval [0, 1].")
        if step_size <= 0:
            raise ValueError("step_size must be positive.")

        defaults = {
            "mutation_rate": mutation_rate,
            "step_size": step_size,
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

    def step(self, closure: Callable[[], torch.Tensor]) -> torch.Tensor:
        if closure is None:
            raise ValueError("GA requires a closure that returns the loss.")

        if not self._initialized:
            loss_tensor = self._evaluate_loss(closure)
            loss = float(loss_tensor.detach().item())

            self._initialized = True
            self._current_loss = loss
            self._update_best_loss(loss)
            self._best_params = self._clone_all_params()

            return loss_tensor

        current_params = self._clone_params()
        current_loss = self._current_loss

        population = self._initialize_population(current_params)
        population_losses, _ = self._evaluate_population(population, closure)

        selected = self._select_population(population, population_losses)
        children = self._crossover(selected)
        children = self._mutate(children)

        candidate_population = selected + children
        candidate_losses, result_template = self._evaluate_population(
            candidate_population, closure
        )

        best_idx = min(
            range(len(candidate_losses)),
            key=lambda idx: candidate_losses[idx],
        )

        best_candidate = candidate_population[best_idx]
        best_candidate_loss = candidate_losses[best_idx]

        self.proposed_steps += len(candidate_population)

        if best_candidate_loss <= current_loss:
            self._restore_params(best_candidate)
            self.accepted_steps += 1
            self._current_loss = best_candidate_loss

            if self.best_loss is None or best_candidate_loss < self.best_loss:
                self.best_loss = best_candidate_loss
                self._best_params = self._clone_all_params()

            return result_template.detach().new_tensor(best_candidate_loss)

        self._restore_params(current_params)
        self.rejected_steps += 1

        return result_template.detach().new_tensor(current_loss)

    def _evaluate_loss(
        self,
        closure: Callable[[], torch.Tensor],
    ) -> torch.Tensor:
        with torch.enable_grad():
            loss_tensor = closure()

        self._record_eval(float(loss_tensor.detach().item()))
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

                individual.append(param + group["step_size"] * noise)

            population.append(individual)

        return population

    def _evaluate_population(
        self,
        population: list[list[torch.Tensor]],
        closure: Callable[[], torch.Tensor],
    ) -> tuple[list[float], torch.Tensor]:
        original = self._clone_params()
        losses = []

        for individual in population:
            self._restore_params(individual)

            with torch.enable_grad():
                loss_tensor = closure()

            loss = float(loss_tensor.detach().item())
            self._record_eval()
            losses.append(loss)

        self._restore_params(original)
        return losses, loss_tensor

    def _select_population(
        self,
        population: list[list[torch.Tensor]],
        losses: list[float],
    ) -> list[list[torch.Tensor]]:
        keep_count = max(2, self.population_size // 2)

        ranked_indices = sorted(
            range(len(losses)),
            key=lambda idx: losses[idx],
        )

        return [
            self._clone_individual(population[idx])
            for idx in ranked_indices[:keep_count]
        ]

    @torch.no_grad()
    def _crossover(
        self,
        parents: list[list[torch.Tensor]],
    ) -> list[list[torch.Tensor]]:
        child_count = self.population_size - len(parents)
        children = []

        for _ in range(child_count):
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

            parent1 = parents[idx1]
            parent2 = parents[idx2]

            child = []

            for p1, p2 in zip(parent1, parent2):
                mask = self._rand_like(p1) < 0.5

                child_param = torch.where(mask, p1, p2)
                child.append(child_param)

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

                new_param = param + mutation_mask * group["step_size"] * noise
                new_individual.append(new_param)

            mutated.append(new_individual)

        return mutated

    @staticmethod
    def _clone_individual(
        individual: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        return [param.detach().clone() for param in individual]

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
        if "step_size" in param_group and param_group["step_size"] <= 0:
            raise ValueError("step_size must be positive in every parameter group.")

    def _algorithm_checkpoint_state(self) -> dict:
        return {"population_size": self.population_size}

    def _load_algorithm_checkpoint_state(self, state: dict) -> None:
        population_size = state["population_size"]
        if population_size < 2:
            raise ValueError("Checkpoint population_size must be at least 2.")
        self.population_size = population_size

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
