# PyPerch agent guide

PyPerch provides lightweight randomized optimizers for ordinary PyTorch models,
plus a thin Optuna search layer for optional hyperparameter studies.

## PyTorch first

- Do not get in the way of PyTorch. Optimizers accept native parameter iterables,
  including `model.parameters()`, and fit ordinary PyTorch training patterns.
- PyPerch owns randomized optimizer behavior and the small amount of optimizer
  state needed to operate on PyTorch parameters. It does not own model
  architecture, tensors, data loading, losses, forward passes, training loops,
  `torch.no_grad()`, parameter freezing, device handling, trainer frameworks,
  callbacks, configuration DSLs, model wrappers, sklearn-style estimators, or a
  replacement experiment framework. Use PyTorch abstractions for those concerns.
- Keep Optuna native and visible. The search layer may coordinate a study, but it
  must not hide Optuna trials, studies, or PyTorch training code.

## Repository map

| Path | Purpose |
| --- | --- |
| `pyperch/optim/` | PyTorch optimizer base and RHC, SA, and GA implementations |
| `pyperch/search/` | Thin Optuna search support and small search utilities |
| `tests/` | Focused optimizer and search tests |
| `examples/standalone/` | Runnable PyTorch optimizer examples |
| `examples/search/` | Runnable Optuna search example |
| `docs/` | General optimizer and search usage guides |
| `pyproject.toml` | Supported Python versions, dependencies, and tool settings |

## Validation

Keep changes small and follow neighboring implementations, tests, examples, and
documentation. The repository checks used by CI are:

```bash
poetry run ruff format --check .
poetry run ruff check .
poetry run pytest
```

Run the checks relevant to the change, and run all three for runtime changes.
Documentation-only work must still validate links, referenced paths and commands,
scope, and the complete diff.

## Security

- Never commit, expose, print, log, or share secrets. Do not embed credentials in
  source, tests, examples, docs, prompts, or committed configuration.
- Never commit `.env` or similar local credential files, expose authenticated URLs,
  private endpoints, or personal filesystem paths, or weaken existing ignore rules.
- Use fake placeholders in examples and secure configuration such as environment
  variables when credentials are required.
- Do not add telemetry, network calls, downloads, or dependencies without a project
  requirement.
- Before handoff, inspect the final diff for credentials, sensitive information,
  generated artifacts, and local files.

## Skill routing

| Task | Read |
| --- | --- |
| Add or change an optimizer | [`.agents/skills/add-or-modify-optimizer/SKILL.md`](.agents/skills/add-or-modify-optimizer/SKILL.md) |
| Other work | This guide and the nearest implementation, test, example, and documentation |

## Project documentation

- [General Usage Guide](docs/general_usage_guide.md)
- [Optuna Search Usage Guide](docs/search.md)
- [Runnable examples](examples/standalone/)

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
