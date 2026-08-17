from __future__ import annotations

import sys


def main() -> int:
    """Entrypoint for the `radarsim` console script.

    The simulation engine isn't implemented yet (see docs/ARCHITECTURE.md
    "Phased build order") — this placeholder just confirms the package
    installs and the entrypoint wires up correctly.
    """
    print("radarsim: CLI not yet implemented. See docs/ARCHITECTURE.md for the build plan.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
