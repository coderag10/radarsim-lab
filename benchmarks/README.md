# benchmarks

Standalone performance scripts for computationally expensive
algorithms (FFT/signal processing, detection, tracking filters).
These are run manually, not as part of `pytest`/CI — they measure
wall-clock/throughput, not correctness.

Populated starting in the phase that introduces each algorithm (see
`../docs/ARCHITECTURE.md` "Phased build order"). Expected files:

- `benchmark_fft.py`
- `benchmark_detection.py`
- `benchmark_tracking.py`
