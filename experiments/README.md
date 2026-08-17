# experiments

- `notebooks/` — Jupyter notebooks for exploratory analysis (e.g.
  `detection.ipynb`, `tracking.ipynb`, `filters.ipynb`), added as each
  phase lands.
- `configs/` — experiment configuration files (e.g. filter-comparison
  sweeps) distinct from `../scenarios/` — a scenario defines the
  world, an experiment config defines what to run against it and how.
- `results/` — generated experiment output (git-ignored; not source).

Install the `notebooks` extra to work here: `uv sync --extra notebooks`.
