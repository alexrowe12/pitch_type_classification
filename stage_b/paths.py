"""Shared filesystem paths for Stage B scripts."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
STAGE_B_DIR = DATA_DIR / "stage_b"

FRAMES_DIR = STAGE_B_DIR / "frames"
LABELS_DIR = STAGE_B_DIR / "labels"
SEQUENCES_DIR = STAGE_B_DIR / "sequences"
DEBUG_DIR = STAGE_B_DIR / "debug"
REVIEW_DIR = STAGE_B_DIR / "review"

FRAME_EXPORTS_CSV = LABELS_DIR / "frame_exports.csv"
WEAK_EVENTS_CSV = LABELS_DIR / "weak_events.csv"
MANUAL_EVENTS_CSV = LABELS_DIR / "manual_events.csv"
FINAL_EVENTS_CSV = LABELS_DIR / "final_events.csv"

ALL_STAGE_B_DIRS = [
    STAGE_B_DIR,
    FRAMES_DIR,
    LABELS_DIR,
    SEQUENCES_DIR,
    DEBUG_DIR,
    REVIEW_DIR,
]


def ensure_stage_b_dirs() -> None:
    """Create the Stage B directory tree if it does not already exist."""
    for path in ALL_STAGE_B_DIRS:
        path.mkdir(parents=True, exist_ok=True)
