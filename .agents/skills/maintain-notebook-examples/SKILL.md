---
name: maintain-notebook-examples
description: Create or revise PyPerch executed teaching notebooks and example guidance while keeping native PyTorch usage, terminology, saved outputs, and plots accurate. Use for notebook behavior, public API, optimizer semantics, convergence, device, or example-presentation changes, not ordinary optimizer development or standalone experiments.
---

# Maintain notebook examples

## When to use

Use this skill for `examples/notebooks/`, their index, or behavior that changes what
those notebooks teach. Follow [CONTRIBUTING.md](../../../CONTRIBUTING.md) for setup and
validation commands.

## Read first

- Read [AGENTS.md](../../../AGENTS.md), [examples/README.md](../../../examples/README.md),
  the affected notebook, and the source, tests, and usage guide for every demonstrated
  API.
- Read [docs/plotting.md](../../../docs/plotting.md) for figures and
  [docs/search.md](../../../docs/search.md) for Optuna examples.

## Invariants

- Keep the teaching path short and concept-led. Show ordinary PyTorch modules,
  parameter iterables, closures, loops, freezing, devices, and evaluation directly.
- Use current API names and precise terminology. Distinguish optimizer calls, RHC or
  SA proposals, GA generations, objective evaluations, monitoring evaluations,
  validation, and test evaluation.
- Demonstrate `restore_best()` before a reported result, comparison, or final
  evaluation that claims the optimizer's best-found state. Preserve current
  parameters when the intended claim is terminal-state performance.
- Use relevant available devices for executable coverage and state unavailable-device
  coverage explicitly. Treat accelerator runs as compatibility evidence unless they
  are designed as performance benchmarks.
- Keep plotting composable with caller-owned Matplotlib Axes. Learning curves use
  training sample counts, training curves label actual iteration, generation, or
  evaluation units, and model-complexity validation curves vary a real capacity
  hyperparameter.
- Keep Optuna studies and trials visible. Refit the selected best configuration and,
  when its evaluation claims the optimizer's best-found state, restore its best
  parameters before evaluating it. Preserve clear optional dependency behavior.

## Workflow

1. Identify the smallest affected notebook set and update code, narrative, labels,
   saved outputs, and example index together.
2. When notebook behavior or output, optimizer semantics, public API, or convergence
   claims change, execute affected notebooks from fresh kernels using the workflow in
   `CONTRIBUTING.md`. Run the full set when shared behavior can affect all notebooks.
3. Check qualitative convergence and compare printed counters with the described
   units. Visualize regenerated plots and inspect titles, axes, legends, uncertainty,
   alt text, spacing, clipping, and readability.
4. Keep temporary execution reports, HTML exports, caches, and local images out of
   the patch unless the repository intentionally tracks that output.

## Validation

- Begin with the smallest affected notebook or example, then perform the broader
  validation required by `CONTRIBUTING.md` for the touched behavior.
- Confirm a fresh run succeeds without hidden interactive state, generated downloads,
  private endpoints, personal paths, or undeclared dependencies.
- Inspect notebook JSON only as a storage format; validate behavior by execution and
  figures by visual review.

## Common failure modes

- Saving output from a warm kernel that no longer matches the code cells.
- Renaming an API without updating prose, labels, counters, and neighboring notebooks.
- Calling GA initialization a generation or presenting monitoring calls as optimizer
  evaluations.
- Leaving execution reports, exports, caches, or ad hoc plots in the diff.
