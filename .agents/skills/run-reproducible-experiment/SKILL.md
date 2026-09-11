---
name: run-reproducible-experiment
description: Run a comparative PyPerch optimizer experiment with a frozen implementation, explicit budgets, fixed seeds and settings, native PyTorch and Optuna use, and reproducible evidence. Use for comparisons or performance claims, not optimizer development or generic training.
---

# Run a reproducible experiment

## When to use

Use this skill for comparisons among optimizer implementations or settings and for
performance claims. Follow [CONTRIBUTING.md](../../../CONTRIBUTING.md) to validate the
implementation before freezing it.

## Read first

- Read [AGENTS.md](../../../AGENTS.md), relevant optimizer source and tests,
  [docs/general_usage_guide.md](../../../docs/general_usage_guide.md), and the affected
  examples.
- Read [docs/search.md](../../../docs/search.md) when Optuna is involved and
  [docs/plotting.md](../../../docs/plotting.md) when preparing figures.
- State the question, baseline, intended comparison, and claim the evidence could
  support before running anything.

## Invariants

- Freeze one validated implementation for all runs. Record the repository state,
  dependency versions, hardware, command, warnings, failures, and limitations.
- Fix and report seeds plus all comparison settings: data split and preprocessing,
  model and initialization, loss, metric, optimizer configuration, restoration,
  budgets, repeats, threading, device, search space, sampler, direction, and trial
  count. Change only the declared comparison factor.
- Use native PyTorch tensors, modules, losses, parameter iterables, evaluation, and
  loops. Keep Optuna `Study` and `Trial` objects visible, retain every trial state,
  and preserve documented optional-dependency behavior.
- Tune and select with validation data. Keep test data untouched until final
  evaluation. Refit the selected configuration. When a result, comparison, or final
  evaluation claims the optimizer's best-found state, call `restore_best()` as the
  optimizer contract requires before evaluating it.

## Measurement units

Record these separately rather than treating them as interchangeable:

- Optimizer step calls are explicit `step()` invocations. For RHC and SA, the first call
  initializes the loss and later calls propose moves.
- GA population initialization is not a generation. Each later `step()` evolves one
  persistent generation, and cached elite fitness avoids some objective calls.
- Objective or function evaluations are actual optimizer closure calls, as reported
  by `function_evals`.
- Monitoring-only evaluations are extra forward evaluations used to record curves;
  they are outside `function_evals` and must be reported separately.
- Validation and test evaluations are also outside the optimizer budget and serve
  selection and final assessment respectively.

Equal optimizer-call counts do not imply equal objective calls or compute. Prefer a
budget that matches the claim, report the other units, and include backward work,
synchronization, or other excluded costs explicitly.

## Workflow

1. Write the fixed protocol and deterministic per-run or per-trial seed rule before
   the first comparison.
2. Keep runners, studies, logs, and results in a task-authorized untracked location
   unless tracked artifacts are requested. Do not change the implementation after
   measurements begin.
3. Record per-run results and all completed, failed, and pruned trials. Compare only
   protocols that differ by the declared factor.
4. Present variation across seeds or splits and label exploratory evidence. Do not
   infer causal improvement from jointly tuned values, tied validation results, or
   a small sample.

## Accelerator claims

Treat CUDA or MPS coverage as compatibility and correctness unless the protocol
actually benchmarks performance. Small randomized-optimization workloads can be
slower on accelerators because of synchronization, kernel launch, transfer, and
state-management overhead. Report timing methodology and warmup when making a speed
claim.

## Validation

Scale validation rigor with the strength of the claim.

- Confirm the frozen implementation passes the concern-specific and repository
  checks required by `CONTRIBUTING.md`.
- Re-run a small deterministic slice before the full protocol and verify that
  counters, restoration, seeds, recorded settings, and evaluation units are correct.
- Inspect plots against `docs/plotting.md`; axes must name the actual iteration,
  generation, sample-count, hyperparameter, or measured-evaluation unit.

## Common failure modes

- Comparing GA generations with RHC or SA proposals as though they cost the same.
- Counting curve monitoring, validation, or test passes as optimizer function calls.
- Reporting accelerator compatibility as a speedup without a benchmark.
- Selecting from test results or claiming best-found performance from the last trial
  or an unrestored final state.
