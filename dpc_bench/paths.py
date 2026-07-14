"""Default paths."""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
DEFAULT_SEED_DIR = REPO_ROOT / "data" / "seed"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output"
