from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from radarsim.cli.run import format_json, format_table, run_scenario

_DEFAULT_NOISE_STD = (0.3, 0.1, 0.01)
_DEFAULT_REFERENCE_RCS = 5.0
_DEFAULT_REFERENCE_RANGE = 30.0
_DEFAULT_REFERENCE_SNR_DB = 20.0
_DEFAULT_SNR_THRESHOLD_DB = 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="radarsim",
        description=(
            "Run a radar simulation scenario end-to-end: target motion, radar "
            "measurement, detection, tracking, and metrics."
        ),
    )
    parser.add_argument("scenario", type=Path, help="Path to a scenario YAML file")
    parser.add_argument(
        "--radar-x", type=float, default=0.0, help="Sensor X position (default: 0.0)"
    )
    parser.add_argument(
        "--radar-y", type=float, default=0.0, help="Sensor Y position (default: 0.0)"
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        nargs=3,
        metavar=("RANGE", "RADIAL_VELOCITY", "ANGLE"),
        default=_DEFAULT_NOISE_STD,
        help="Measurement noise std: range, radial velocity, angle (default: 0.3 0.1 0.01)",
    )
    parser.add_argument(
        "--reference-rcs",
        type=float,
        default=_DEFAULT_REFERENCE_RCS,
        help="RCS (m^2) used to calibrate the radar equation constant (default: 5.0)",
    )
    parser.add_argument(
        "--reference-range",
        type=float,
        default=_DEFAULT_REFERENCE_RANGE,
        help="Range used to calibrate the radar equation constant (default: 30.0)",
    )
    parser.add_argument(
        "--reference-snr-db",
        type=float,
        default=_DEFAULT_REFERENCE_SNR_DB,
        help="Target SNR (dB) at the reference RCS/range (default: 20.0)",
    )
    parser.add_argument(
        "--snr-threshold-db",
        type=float,
        default=_DEFAULT_SNR_THRESHOLD_DB,
        help="Detection SNR threshold in dB (default: 0.0)",
    )
    parser.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
        help="Output format (default: table)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        result = run_scenario(
            args.scenario,
            radar_position=np.array([args.radar_x, args.radar_y]),
            noise_std=np.array(args.noise_std),
            reference_rcs=args.reference_rcs,
            reference_range=args.reference_range,
            reference_snr_db=args.reference_snr_db,
            snr_threshold_db=args.snr_threshold_db,
        )
    except (ValueError, OSError) as error:
        print(f"radarsim: error: {error}", file=sys.stderr)
        return 1

    formatted = format_json(result) if args.format == "json" else format_table(result)
    print(formatted)
    return 0


if __name__ == "__main__":
    sys.exit(main())
