# scenarios

YAML scenario definitions consumed by `radarsim.io.load_scenario`
(simulation duration/timestep/seed, target definitions, radar/sensor
configuration). The schema is finalized in Phase 1 — see
`../docs/ARCHITECTURE.md`.

Subdirectories, populated as each is exercised:

- `basic/` — single target, low noise
- `multi-target/` — multiple targets, data-association stress test
- `high-noise/` — degraded SNR, detection/tracking robustness
- `sensor-failure/` — dropped/degraded sensor, fusion robustness
