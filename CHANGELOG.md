# Changelog

## Unreleased

- Persist GA populations and known elite losses across generations, count GA steps
  in generation units, and preserve population state across checkpoints.
- Replace GA `step_size` with separate `initialization_step_size` and
  `mutation_step_size` settings, add RHC `restart_scale`, and tighten SA temperature
  validation.
- Add `pyperch.plotting` preparation and rendering APIs for learning, training,
  and validation curves. Renderers require caller-owned Matplotlib Axes and return
  the same Axes without managing figures or starting training.
- Add repeated-run mean summaries with sample standard deviation or observed-range
  bands, explicit repeat counts, and no uncertainty band for single runs.
- Replace ten example scripts with four concept-led, executed notebooks covering
  native training, optimizer comparisons, learning and validation curves, and
  Optuna tuning.
- Update documentation with curve terminology, plotting contracts, notebook entry
  points, optional dependencies, and reproducibility guidance.
- Remove the README blurb about agent-assisted experiments.
