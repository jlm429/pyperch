---
name: add-or-modify-optimizer
description: Add or change a PyPerch randomized optimizer while preserving native PyTorch use, local optimizer conventions, and focused evidence. Use for optimizer implementation work, not search-only or documentation-only tasks.
---

# Add or modify an optimizer

Read the repository [agent guide](../../../AGENTS.md) first. This workflow adds the
optimizer-specific detail that does not belong in the always-loaded guide.

## Establish the local contract

1. Read the algorithm carefully enough to identify its state, proposal, acceptance,
   evaluation, and randomization rules before choosing an API.
2. Inspect `pyperch/optim/base.py`, neighboring optimizer implementations,
   `tests/optim/`, the matching `examples/standalone/` directories, and
   `docs/general_usage_guide.md`. Check public exports when adding a class.
3. For a bug fix, reproduce the user-visible failure through a normal PyTorch usage
   path before changing code.
4. Follow local patterns where they express a shared contract, but do not
   mechanically force different algorithms into identical implementations.

## Design the smallest PyTorch-compatible API

- Accept native PyTorch parameter iterables and preserve direct
  `Optimizer(model.parameters(), ...)` usage and ordinary training loops.
- Use closure evaluation when the algorithm needs to reevaluate a loss, following
  the local optimizer pattern. Leave forward passes, loss construction, gradient
  mode, evaluation, freezing, and device choices in normal PyTorch code.
- Add only algorithm-specific options and state. Do not introduce trainers,
  callbacks, model wrappers, estimator APIs, configuration layers, tensor or data
  abstractions, or experiment-framework behavior.
- Touch `pyperch/search/` or Optuna documentation only when the concern genuinely
  belongs to hyperparameter search. Keep native Optuna trials and studies visible.

## Implement consistently

- Respect parameter groups, `requires_grad`, tensor dtype, and tensor device using
  PyTorch operations. Preserve the public behavior of existing constructor defaults.
- Follow the neighboring patterns for current and best parameters, best loss,
  `restore_best()`, and the `function_evals`, `proposed_steps`, `accepted_steps`, and
  `rejected_steps` counters. Adapt those patterns when the algorithm requires a
  different meaning, and document the difference.
- Consider reproducibility explicitly. Define how `random_state` affects proposals,
  initialization, crossover, mutation, or acceptance, and avoid accidental reliance
  on unrelated global random state.
- Keep the patch focused. Record desirable adjacent refactors separately unless they
  are required for correct behavior.

## Supply evidence and usage

- Add focused tests under `tests/optim/` for the algorithm behavior, validation,
  counters and state, reproducibility where relevant, frozen parameters, and backward
  compatibility affected by the change. Prefer observable behavior over private
  implementation assertions.
- Add or update a runnable example under `examples/standalone/<optimizer>/` that uses
  a normal `torch.nn.Module`, native parameters, a loss closure, and an ordinary loop.
- Update `docs/general_usage_guide.md`, public exports, and README links only as the
  public change requires.

## Validate and review

Run the repository checks from `AGENTS.md`, including the focused optimizer tests and
the runnable example when practical. Inspect the complete diff for API or abstraction
creep, duplicated framework behavior, unrelated Optuna changes, stale documentation,
secrets, local files, and generated artifacts.
