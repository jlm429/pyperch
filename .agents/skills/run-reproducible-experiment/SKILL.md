---
name: run-reproducible-experiment
description: Run a comparative PyPerch optimizer experiment with a frozen implementation, fixed protocol, native PyTorch and Optuna use, and reproducible evidence. Use for experiments that compare optimizer settings or implementations, not ordinary optimizer development or generic model training.
---

# Run a reproducible experiment

Read the repository [agent guide](../../../AGENTS.md) first. State the experimental
question, baseline, intended comparison, and claim the evidence could support before
running anything.

## Freeze the comparison

- Validate the implementation first, then freeze it for every run. Record the commit,
  dirty status, relevant diff or file hashes, and dependency versions.
- Give experiment workers read-only access to the implementation. Write runners,
  logs, studies, and results outside the repository unless the task requests tracked
  artifacts.
- Fix the protocol before the first run: data split and preprocessing, model and
  initialization, loss, optimizer calls, restoration of best parameters, metric,
  budget, search space, sampler, direction, trial count, threading, and hardware.
  Change only the intended comparison factor.

## Keep PyTorch and Optuna visible

Use ordinary PyTorch tensors, modules, losses, evaluation, and loops. Instantiate the
PyPerch optimizer directly with native parameter iterables such as
`GA(model.parameters(), ...)`. Account for optimizer semantics that affect the budget,
including initialization-only calls and whether `restore_best()` is required before
evaluation.

When search is needed, use the existing `OptunaSearch` layer without hiding its
native `Study` or `Trial` objects. Record the exact Optuna distributions, including
any approximation required by PyPerch's tuple format, and retain every completed,
failed, and pruned trial state.

## Make comparisons reproducible

- Record every experiment seed and one deterministic per-trial seed rule. Seed each
  random source the protocol uses and pass the trial seed to the PyPerch optimizer.
- Tune and select on validation data. Keep test data untouched until the protocol's
  final evaluation, and never select configurations from test results.
- Record the command, repository snapshot, dirty files, environment, exact protocol,
  trial budget and states, best trial and parameters, objective value, runtime,
  warnings, failures, and limitations.
- Compare runs only when their protocols match apart from the declared factor. Treat
  small searches or inconsistent seed rules as exploratory evidence. Do not infer a
  causal improvement from a jointly tuned parameter, tied validation scores, or a
  small number of seeds and splits.
