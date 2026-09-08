# Changelog

## Unreleased

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
