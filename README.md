# radarsim-lab

A modular, reproducible radar simulation and target-tracking laboratory for studying synthetic radar measurements, signal processing, detection, tracking, sensor fusion, and algorithm performance.

Built as a research platform, not a demo: ground truth, measurements, detections, and track estimates are kept as separate, non-interchangeable types so tracking/detection algorithms can never "cheat" by seeing the truth they're supposed to be estimating. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full pipeline design.

## Status

Early scaffold — repository structure and interfaces only. No simulation logic is implemented yet; it is being built incrementally, one pipeline phase at a time (see `docs/ARCHITECTURE.md` for the phase plan).

## Pipeline

```
Scenario (YAML) → Target Generator → Radar Model → Signal Processing → Detection
                                                                            │
                                                                            ▼
                                          Sensor Fusion ← State Estimator ← Track Manager
                                                │
                                                ▼
                                          Metrics / Visualization
```

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or plain `pip` + `venv`

## Quickstart

Clone the repo, then install and verify the scaffold:

```bash
git clone https://github.com/coderag10/radarsim-lab.git
cd radarsim-lab
```

**With uv (recommended):**

```bash
uv sync --extra dev          # install runtime + dev dependencies into .venv
uv run pytest                # run the test suite
uv run ruff check .          # lint
uv run mypy src               # type-check
uv run radarsim               # smoke-test the CLI entrypoint
```

**With plain pip + venv:**

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
pytest
ruff check .
mypy src
radarsim
```

Everything here is a stub at this stage (see [Status](#status)), so `pytest` currently only verifies the package imports cleanly, and `radarsim` prints a placeholder message — that's expected until Phase 1 lands.

## Repository layout

- `src/radarsim/` — the simulation engine (core, targets, radar, signals, detection, tracking, fusion, metrics, io)
- `src/radarsim/api/`, `src/radarsim/cli/` — service and command-line entrypoints (later phases)
- `dashboard/` — React + TypeScript visualization frontend (later phase)
- `scenarios/` — YAML scenario definitions
- `experiments/` — notebooks, experiment configs, and results
- `tests/` — unit, integration, and regression tests
- `benchmarks/` — performance benchmarks for computationally expensive algorithms

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for module responsibilities and data contracts.
