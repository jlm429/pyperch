---
name: add-or-modify-optimizer
description: Add or change a PyPerch randomized optimizer while preserving native PyTorch use, algorithm semantics, state continuity, and focused evidence. Use for optimizer implementation work, not search-only, notebook-only, or documentation-only tasks.
---

# Add or modify an optimizer

## When to use

Use this skill for behavior, state, or public API changes in `pyperch/optim/`.
Follow [CONTRIBUTING.md](../../../CONTRIBUTING.md) for repository validation commands.

## Read first

- Read [AGENTS.md](../../../AGENTS.md), the claimed algorithm, `pyperch/optim/base.py`,
  neighboring optimizers, relevant `tests/optim/`, and
  [docs/general_usage_guide.md](../../../docs/general_usage_guide.md).
- Read affected notebooks through
  [the notebook skill](../maintain-notebook-examples/SKILL.md) when semantics, public
  API, or convergence examples may change.
- Reproduce a bug through an ordinary end-to-end PyTorch usage path before fixing it.

## Invariants

- Accept native parameter iterables and parameter groups. Keep models, tensors,
  losses, closures, forward passes, training loops, freezing, and devices in normal
  PyTorch code. Add no trainer, wrapper, estimator, callback, configuration, or
  experiment abstraction.
- Keep the implementation recognizable as the claimed algorithm. Define its
  proposal, acceptance, state transition, evaluation, and randomization semantics
  before changing its API.
- Mutate only trainable parameters. Apply group options to their own tensors and
  keep joint-search settings optimizer-wide. Reject invalid or silently ineffective
  settings when constructing or adding groups. Adding a group changes the joint
  search space, so preserve model values and private RNG position while restarting
  run bookkeeping.
- Preserve parameter and returned-loss dtype and device. Use a private RNG for every
  stochastic choice so equal seeds reproduce independently of PyTorch global RNG,
  `None` starts a fresh private stream, and counter resets do not rewind it.
- Count `function_evals` as actual closure calls. Use proposal units for RHC and SA
  step counters and generation units for GA after initialization. Keep
  `accepted_steps + rejected_steps == proposed_steps`.
- Keep `best_loss` and its parameter checkpoint consistent. `restore_best()` restores
  the global best and synchronizes current-loss bookkeeping without rewinding
  counters, schedules, retained population, or RNG state.
- Save enough optimizer state for deterministic continuation with a compatible model
  and group layout. This includes common counters, losses, best parameters, group
  defaults, and RNG state, plus RHC restart progress, SA temperature state, or GA
  population and cached fitness.
- Validate constructor, group, and loaded checkpoint values at the boundary. Avoid
  redundant closure calls and full-model copies when a changed tensor or cached
  fitness is sufficient.

## Algorithm contracts

- RHC accepts non-worsening proposals. Schedule restarts by total proposed steps,
  whether accepted or rejected, preserve the global best across restarts, and treat
  the first evaluation after a restart as an evaluation rather than a proposal.
- SA accepts non-worsening proposals and accepts a worse proposal with Metropolis
  probability `exp(-loss increase / temperature)`. Cool after each proposal and
  respect the configured temperature floor.
- GA initializes one population once, then retains it across calls. Preserve a
  deterministic best-half elite set, with at least one elite and cached fitness.
  Create and evaluate only new children through uniform crossover and group-scaled
  mutation, and make each later call one generation. Keep current model parameters
  aligned with the best current individual and checkpoint the full population and
  its fitness while preserving the global best separately.

## Workflow

1. State the behavior and counter units before implementation, including
   initialization-only calls and all state transitions.
2. Make the smallest source and test change that establishes that behavior.
3. Update public exports, usage documentation, and affected examples only when the
   public contract changes.
4. Inspect the complete diff for abstraction creep, stale semantics, unrelated
   search changes, sensitive data, local artifacts, and generated junk.

## Validation

Scale evidence to touched behavior, then complete the required repository validation
from `CONTRIBUTING.md`.

- RNG or checkpoint work must prove private-RNG isolation and deterministic
  uninterrupted versus resumed continuation.
- Parameter-group work must exercise multiple groups; frozen-parameter work must
  exercise frozen parameters.
- Dtype or device work must cover the relevant supported dtype and every relevant
  available supported device.
- Counter or evaluation work must assert exact units and closure-call counts,
  including initialization and cached evaluations.
- Algorithm-semantic or public API work must run affected examples or notebooks and
  check qualitative convergence, not only unit tests.
- Validation-only changes may begin with focused rejection tests before broader
  required validation.

## Common failure modes

- Using global randomness for one acceptance, crossover, or mutation path.
- Treating initialization or monitoring as a proposal or GA generation.
- Rebuilding a GA population or reevaluating known elites on every call.
- Restoring parameter values without synchronizing cached current loss.
- Copying frozen parameters into proposals or losing algorithm state at checkpoint.
