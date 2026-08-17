# radarsim-lab

A modular, reproducible radar simulation and target-tracking laboratory for studying synthetic radar measurements, signal processing, detection, tracking, sensor fusion, and algorithm performance.

Built as a research platform, not a demo: ground truth, measurements, detections, and track estimates are kept as separate, non-interchangeable types so tracking/detection algorithms can never "cheat" by seeing the truth they're supposed to be estimating. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full pipeline design.

## Status

The full simulation pipeline is real and working end-to-end: **Scenario → Target Generator → Radar Model → Detection → Tracking (Kalman / EKF / particle filter) → Sensor Fusion → Metrics**, all built incrementally, one phase at a time, each with its own tests (see `docs/ARCHITECTURE.md` for the phase-by-phase build log).

- **Done:** `core`, `targets`, `io` (Phase 1) · `radar` (Phase 2) · `signals` (Phase 3, standalone toolkit) · `detection` (Phase 4) · `tracking` — Kalman filter + association + `Tracker` (Phase 5a), EKF + particle filter (Phase 5b) · `metrics` (Phase 6) · `fusion` (Phase 7) · `cli` (Phase 8a) — run `radarsim <scenario.yaml>` to try it · `api` (Phase 8b) — `POST /runs` etc., see [`src/radarsim/api/README.md`](src/radarsim/api/README.md)
- **Not yet built:** `dashboard` (Phase 8c, React frontend)

Run `uv run radarsim scenarios/basic/two_targets.yaml` for a working example (see [Usage](#usage) below).

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

Clone the repo, then install and verify the setup:

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
uv run radarsim --help        # smoke-test the CLI entrypoint
```

`api` (FastAPI) is a separate, optional extra — `uv sync --extra dev --extra api` to also install and test it (see [API](#api) below). Everything degrades gracefully without it: API tests are skipped rather than failed, and `mypy`/`ruff` stay clean either way.

**With plain pip + venv:**

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
pytest
ruff check .
mypy src
radarsim --help
```

## Usage

```bash
uv run radarsim scenarios/basic/two_targets.yaml
```

```
Scenario: scenarios/basic/two_targets.yaml
  Duration: 5.0s  Timestep: 1.0s  Seed: 42  Targets: 2

Sensor: radar-1 @ (0.0, 0.0)

Tracking results (t=5.0s):
  ID        Status    Position            Velocity
  track-0   ACTIVE    (20.03, 25.15)      (1.87, -1.00)
  track-1   ACTIVE    (-15.10, 25.02)     (1.07, 2.07)

Metrics:
  Detection probability: 100.0%
  Position RMSE:          0.130
```

Add `--format json` for machine-readable output, or see `radarsim --help` for sensor position/noise/detection-threshold flags.

## API

`POST /runs` is a thin JSON wrapper around the same pipeline the CLI uses:

```bash
uv sync --extra dev --extra api
uvicorn radarsim.api.main:app --reload
```

```bash
curl -X POST http://127.0.0.1:8000/runs \
  -H "Content-Type: application/json" \
  -d '{"scenario": "scenarios/basic/two_targets.yaml"}'
```

Interactive docs (Swagger UI) at `http://127.0.0.1:8000/docs` once the server is running. See [`src/radarsim/api/README.md`](src/radarsim/api/README.md) for the full endpoint list.

## Repository layout

- `src/radarsim/` — the simulation engine: `core`, `targets`, `radar`, `signals`, `detection`, `tracking`, `fusion`, `metrics`, `io`, `types` (all implemented)
- `src/radarsim/cli/` — command-line entrypoint (implemented, see [Usage](#usage))
- `src/radarsim/api/` — FastAPI service exposing simulation runs (implemented, see [API](#api))
- `dashboard/` — React + TypeScript visualization frontend (not yet built, Phase 8c)
- `scenarios/` — YAML scenario definitions
- `experiments/` — notebooks, experiment configs, and results
- `tests/` — unit, integration, and regression tests (150+ tests across every implemented module; API tests skip gracefully without the `api` extra)
- `benchmarks/` — performance benchmarks for computationally expensive algorithms

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for module responsibilities, data contracts, and the phase-by-phase build log.
