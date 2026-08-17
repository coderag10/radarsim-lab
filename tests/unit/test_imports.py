"""Sanity check: every package in the scaffold imports cleanly.

This is intentionally the only real test at this stage — the modules
it imports are interface stubs (see docs/ARCHITECTURE.md), not
implementations. Once a phase implements a module, its real behavior
gets its own test file alongside this one.
"""

import importlib

import pytest

PACKAGES = [
    "radarsim",
    "radarsim.types",
    "radarsim.core",
    "radarsim.targets",
    "radarsim.radar",
    "radarsim.signals",
    "radarsim.detection",
    "radarsim.tracking",
    "radarsim.tracking.filters",
    "radarsim.fusion",
    "radarsim.metrics",
    "radarsim.io",
    "radarsim.api",
    "radarsim.cli",
]


@pytest.mark.parametrize("package", PACKAGES)
def test_package_imports(package: str) -> None:
    importlib.import_module(package)
