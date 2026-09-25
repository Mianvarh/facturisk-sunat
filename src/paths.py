"""Portable path helpers for the FactuRisk SUNAT project.

The project can run in two modes:
- from source code with Python;
- from a PyInstaller onedir executable.

Writable files must live next to the executable/project root, never inside
PyInstaller's temporary extraction directory.
"""

from __future__ import annotations

import sys
from pathlib import Path


def get_application_root() -> Path:
    """Return the project/executable root directory."""

    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def get_resource_path(*parts: str) -> Path:
    """Return a path to a bundled or source resource."""

    return get_application_root().joinpath(*parts)


def get_writable_path(*parts: str) -> Path:
    """Return a path intended for outputs, logs, data or config files."""

    return get_application_root().joinpath(*parts)


def ensure_directories() -> None:
    """Create the standard writable project directories."""

    root = get_application_root()
    for relative in [
        "config",
        "data/raw",
        "data/raw/sunat",
        "data/raw/sunat/extraido",
        "data/processed",
        "data/models",
        "outputs",
        "logs",
        "docs",
    ]:
        root.joinpath(relative).mkdir(parents=True, exist_ok=True)
