# Architecture

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

Each arrow is a typed contract (see [Data contracts](#data-contracts)), not a shared mutable object. A stage only receives what the previous stage is allowed to produce.

## The one rule that matters

**Ground truth, measurements, detections, and track estimates are separate types.** A target's true `GroundTruth` state exists only inside the simulation (`targets`, `core`). The moment it's "sensed" it becomes a `RadarMeasurement` — noisy, partial, and that's all any downstream algorithm (`detection`, `tracking`, `fusion`) ever sees or is allowed to import. Only `metrics` is permitted to hold both a `TrackEstimate` and its corresponding `GroundTruth` side by side, because comparing them is literally its job (position RMSE, detection probability, false-alarm rate).

If an algorithm module needs to import `radarsim.types.state.GroundTruth`, that's a design smell — it means the algorithm is being handed the answer it's supposed to be estimating.

## Module responsibilities

| Module | Responsibility | Consumes | Produces | Must NOT import |
|---|---|---|---|---|
| `types` | Shared, frozen data contracts | — | — | any algorithm module |
| `core` | Simulation clock, seeded RNG, scenario orchestration/world state | scenario config | `GroundTruth` stream | `radar`, `detection`, `tracking` internals |
| `targets` | Motion models (constant velocity/acceleration, ...), RCS models | scenario config | `GroundTruth` | `radar`, `signals` |
| `radar` | True state → noisy synthetic measurement | `GroundTruth` | `RadarMeasurement` | `tracking`, `fusion` |
| `signals` | Sampling, FFT, noise models, filtering, range-Doppler representation | raw/sampled signal | processed signal / range-Doppler map | `core`, `tracking` |
| `detection` | Signal / measurement → candidate detection with confidence & SNR | `RadarMeasurement` or processed signal | `Detection` | `tracking`, `GroundTruth` |
| `tracking` | Prediction, data association, state estimation (Kalman / EKF / particle filters) | `Detection` stream | `TrackEstimate` | `GroundTruth` |
| `fusion` | Combine tracks/detections across multiple sensors | multiple `TrackEstimate`/`Detection` streams | fused `TrackEstimate` | — |
| `metrics` | Position RMSE (optimal matching), detection probability, false-alarm rate | `TrackEstimate` **and** `GroundTruth` | scalar metrics | — |
| `io` | Scenario YAML loading/validation, run (de)serialization | YAML/JSON | scenario config, `types` objects | algorithm internals |
| `cli` | Command-line entrypoint — run a scenario end-to-end, print a table or JSON summary | argv | stdout / exit code | — |
| `api` | FastAPI service exposing simulation runs, wrapping `cli.run_scenario` | HTTP requests | JSON | — |

## Data contracts

Defined in `src/radarsim/types/`. Frozen, slotted dataclasses backed by NumPy arrays — no validation-heavy framework in the hot path. `pydantic` is reserved for the `api` boundary, where it's actually useful (request/response validation), not for the simulation inner loop.

```python
GroundTruth
    timestamp: float
    target_id: str
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    rcs: float                  # radar cross-section

RadarMeasurement
    timestamp: float
    sensor_id: str
    range: float
    radial_velocity: float
    angle: float
    covariance: np.ndarray      # measurement noise covariance

Detection
    timestamp: float
    measurement: RadarMeasurement
    confidence: float
    snr: float

TrackEstimate
    track_id: str
    timestamp: float
    state: np.ndarray           # filter state vector (e.g. [x, y, vx, vy])
    covariance: np.ndarray
    status: TrackStatus         # TENTATIVE | ACTIVE | LOST
```

## Determinism

Every simulation run is driven by a single seeded RNG owned by `core.rng` and threaded explicitly through target motion, radar noise, and any stochastic filter (particle filter). No module calls `numpy.random` global state directly — this is what makes regression tests (`tests/regression/`) meaningful: the same seed must always produce the same run.

## Testing strategy

- `tests/unit/` — one test module per source module, testing algorithms in isolation against synthetic inputs
- `tests/integration/` — pipeline slices wired together end-to-end (e.g. scenario → measurements → detections → tracks)
- `tests/regression/` — fixed-seed scenarios with golden-value assertions, catching accidental behavior changes
- `hypothesis`-based property tests where an algorithm has an invariant worth stating (e.g. a constant-velocity model's velocity is conserved; a Kalman filter's covariance stays positive semi-definite)
- `benchmarks/` — standalone scripts (outside pytest/CI) for profiling FFT, detection, and tracking hot paths

## Phased build order

Each phase is proposed, reviewed, and implemented separately (one plan, one commit, its own tests). Large phases get split (5 and 8 below) rather than bundled, the same way the whole project is built incrementally rather than all at once.

1. ✅ **Simulation core** — `core` (clock, RNG, world), `targets` (motion models), `io` (scenario loading)
2. ✅ **Radar model** — `radar` (true state → noisy measurement)
3. ✅ **Signal processing** — `signals` (sampling, noise, FFT, filters, range-Doppler). Standalone toolkit, not wired into `radar` (see [Signal processing vs. radar](#signal-processing-vs-radar) below)
4. ✅ **Detection** — `detection` (thresholds, confidence, SNR). Required retroactively extending `radar`/`types` with an `snr` field once it became clear a threshold detector needs a real signal-strength quantity to threshold on
5. **Tracking** — `tracking` (prediction, association, Kalman/EKF/particle filters), split in two:
   - ✅ **5a** — `KalmanFilter` + `NearestNeighbor` association + a new `Tracker` orchestrator (not originally stubbed, but required to run anything). Operates in Cartesian state via a polar→Cartesian "converted measurement" step
   - ✅ **5b** — `ExtendedKalmanFilter` (operates directly on polar measurements, no conversion) + `ParticleFilter`. Neither is wired into `Tracker` yet — see [Filters not wired into Tracker](#filters-not-wired-into-tracker)
6. ✅ **Ground truth & metrics** — `metrics` (RMSE via optimal bipartite matching, detection probability, false-alarm rate). "Track continuity" (ID-switch rate) from the original scope note is deliberately deferred — it needs multi-timestep track-to-truth association, a bigger design than the other three metrics
7. ✅ **Sensor fusion** — `fusion` (`fuse_estimates`, `match_tracks`). Building its integration test exposed and fixed a real bug: `polar_to_cartesian_measurement` had silently assumed the sensor sat at the world origin (see the Phase 7 commit)
8. **Dashboard & API** — `api`, `cli`, `dashboard/`, split in three:
   - ✅ **8a** — `cli`: `radarsim <scenario.yaml>` runs the full pipeline and prints a table or JSON summary
   - ✅ **8b** — `api`: FastAPI service (`GET /scenarios`, `GET /scenarios/{path}`, `POST /runs`, `GET /health`). `POST /runs` is a thin JSON wrapper around `cli.run.run_scenario` — no pipeline logic duplicated. REST only; WebSocket streaming deliberately deferred to when 8c's dashboard exists to define what shape it actually needs, rather than guessing at a protocol now. See [`api` as an optional extra](#api-as-an-optional-extra) below
   - ⬜ **8c** — `dashboard/`: React + TypeScript frontend

## Signal processing vs. radar

`signals` (Phase 3) is a standalone sample-domain toolkit — FFT, noise, filters, range-Doppler maps — deliberately **not** wired into `radar.RadarModel`, which produces measurements analytically (range/angle/radial-velocity computed directly from geometry, not from a simulated waveform). Connecting them would mean simulating actual chirp/pulse waveforms, a materially larger project than anything built so far.

## Filters not wired into Tracker

`Tracker` (Phase 5a) only accepts a `KalmanFilter` and hardcodes the KF-specific polar→Cartesian measurement conversion. `ExtendedKalmanFilter` and `ParticleFilter` (Phase 5b) are validated standalone instead:
- EKF against the real two-target scenario, operating directly on polar measurements (a manual predict/update loop, not through `Tracker`)
- The particle filter against a synthetic linear-Gaussian problem, where its estimate should converge to the same posterior a Kalman filter computes exactly — and does, within 0.01 position / 0.001 velocity using 5000 particles

Making `Tracker` filter-agnostic (an injected measurement-adapter alongside the filter) and giving `ParticleFilter` a per-track home in `Tracker`'s one-shared-filter design are both legitimate future work, not attempted yet.

## `api` as an optional extra

`api` (Phase 8b) depends on `fastapi`/`pydantic`/`uvicorn`, installed via `uv sync --extra api` — kept separate from `dev` rather than folded in, since it's a genuinely optional piece (someone hacking on the simulation engine shouldn't need a web framework installed). Two things make that split actually work rather than just being aspirational:
- `tests/unit/test_api_*.py` all start with `pytest.importorskip("fastapi")`, so `uv run pytest` with only `--extra dev` skips them (not fails) — the standard verification command every phase has used stays valid.
- `pyproject.toml`'s `[[tool.mypy.overrides]]` treats `fastapi`/`pydantic`/`uvicorn` imports as `Any` when unresolved (`ignore_missing_imports`), and separately relaxes `disallow_subclassing_any`/`disallow_untyped_decorators` *only* for `radarsim.api.*` — both are needed together: the first stops mypy erroring on the missing import itself, the second stops `strict`'s other rules from still hard-failing once `BaseModel`/route decorators resolve to `Any`. Installing `--extra api` gets full strict checking against the real stubs either way.
