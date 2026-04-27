"""Shared filesystem paths for binary modeling scripts."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELING_DIR = DATA_DIR / "modeling"

VARIANTS_DIR = MODELING_DIR / "variants"
RUNS_DIR = MODELING_DIR / "runs"
DEBUG_DIR = MODELING_DIR / "debug"

ALL_MODELING_DIRS = [
    MODELING_DIR,
    VARIANTS_DIR,
    RUNS_DIR,
    DEBUG_DIR,
]


def ensure_modeling_dirs() -> None:
    """Create the modeling directory tree if it does not already exist."""
    for path in ALL_MODELING_DIRS:
        path.mkdir(parents=True, exist_ok=True)
