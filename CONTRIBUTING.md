# Contributing to PyPerch

PyPerch is a small research and teaching library. Contributions should keep
randomized optimizers usable with ordinary PyTorch models, parameter iterables,
losses, and training loops. Prefer focused changes over new abstraction layers.

## Before changing code

- Read `AGENTS.md` and the nearest implementation, tests, example, and guide.
- Check the worktree for unrelated changes and keep the patch reviewable.
- Reproduce bugs through a normal end-to-end PyTorch usage path before fixing them.
- Confirm that the concern belongs in PyPerch optimizer behavior or its thin Optuna
  layer. Models, data, evaluation, devices, and training loops remain PyTorch code.
- Avoid new dependencies unless the project requirement clearly needs one.

PyPerch supports Python 3.10 through 3.13 and PyTorch 2.1 up to, but not including,
3.0. `poetry install` installs the current runtime dependencies, including Optuna,
and the development group with pytest, Ruff, scikit-learn, and Matplotlib. CircleCI
installs that dependency set and runs on Python 3.12.

## Validation

CircleCI runs these repository checks:

```bash
poetry run ruff format --check .
poetry run ruff check .
poetry run pytest
```

Run focused tests and relevant examples while developing, then run all three checks
for runtime changes. For documentation-only changes, validate links, referenced
paths and commands, scope, and the complete diff. Tests are necessary evidence, but
they do not by themselves establish algorithm correctness, API quality, backward
compatibility, or performance claims.

## Change expectations

- Runtime changes need focused tests for observable behavior and documentation when
  public behavior changes.
- Optimizer work should follow the
  [optimizer skill](.agents/skills/add-or-modify-optimizer/SKILL.md), preserve native
  `model.parameters()` use, and avoid trainers, model wrappers, estimator APIs, or
  configuration frameworks.
- Examples should be runnable ordinary PyTorch programs. Keep data preparation,
  models, losses, evaluation, freezing, and device handling visible.
- Compatibility changes should account for the declared Python and PyTorch ranges,
  constructor defaults, public imports, parameter dtype and device, and frozen
  parameters where relevant.
- Experiment or performance claims need a fixed comparison protocol, reproducible
  evidence, limitations, and a clear distinction between validation and test data.

Keep the README focused on installation, user entry points, and links. Put detailed
optimizer usage in `docs/general_usage_guide.md`, Optuna usage in `docs/search.md`,
and executable teaching material in `examples/`. Keep `AGENTS.md` as the concise
repository map and invariant set, task-specific agent guidance in `.agents/skills/`,
and `CLAUDE.md` as a pointer to `AGENTS.md`.

## Agent-assisted contributions

Coding agents are welcome. Contributors remain accountable for the resulting design,
code, claims, and review. Inspect the full diff, verify agent-produced evidence, and
call out material limitations or unresolved concerns. Do not treat an automated
review or green test suite as a substitute for understanding the algorithm and API.

## Security

Never commit or expose keys, tokens, credentials, `.env` files, authenticated or
private URLs and endpoints, personal filesystem paths, or sensitive local experiment
artifacts. Do not add telemetry, network calls, or downloads without a project
requirement. Before submitting, inspect the diff and status for secrets, generated
files, local artifacts, and unrelated changes.

## Pull requests

Keep each pull request focused. Explain the problem and approach, list the validation
performed, and identify compatibility effects, limitations, or follow-up work that a
reviewer should understand. Link supporting evidence for algorithm or performance
claims without overstating what it proves.
