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

**Ground truth, measurements, detections, and track estimates are separate types.** A target's true `GroundTruth` state exists only inside the simulation (`targets`, `core`). The moment it's "sensed" it becomes a `RadarMeasurement` — noisy, partial, and that's all any downstream algorithm (`detection`, `tracking`, `fusion`) ever sees or is allowed to import. Only `metrics` is permitted to hold both a `TrackEstimate` and its corresponding `GroundTruth` side by side, because comparing them is literally its job (RMSE, detection probability, false-alarm rate, track continuity).

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
| `metrics` | RMSE, detection probability, false-alarm rate, precision/recall, latency | `TrackEstimate` **and** `GroundTruth` | scalar/series metrics | — |
| `io` | Scenario YAML loading/validation, run (de)serialization | YAML/JSON | scenario config, `types` objects | algorithm internals |
| `api` | FastAPI service exposing simulation runs (Phase 7) | HTTP requests | JSON | — |
| `cli` | Command-line entrypoint | argv | stdout / files | — |

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

1. **Simulation core** — `core` (clock, RNG, world), `targets` (motion models), `io` (scenario loading)
2. **Radar model** — `radar` (true state → noisy measurement)
3. **Signal processing** — `signals` (sampling, noise, FFT, filters, range-Doppler)
4. **Detection** — `detection` (thresholds, confidence, SNR)
5. **Tracking** — `tracking` (prediction, association, Kalman/EKF/particle filters)
6. **Ground truth & metrics** — `metrics` (RMSE, Pd, FAR, track continuity) plus multi-target association algorithms
7. **Sensor fusion** — `fusion`
8. **Dashboard & API** — `api`, `cli`, `dashboard/`

Each phase is proposed, reviewed, and implemented separately — see the repository's plan history rather than assuming everything above is already built.
