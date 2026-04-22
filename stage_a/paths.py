from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
STAGE_A_DIR = DATA_DIR / "stage_a"

FRAMES_DIR = STAGE_A_DIR / "frames"
EMBEDDINGS_DIR = STAGE_A_DIR / "embeddings"
LABELS_DIR = STAGE_A_DIR / "labels"
MODELS_DIR = STAGE_A_DIR / "models"
PREDICTIONS_DIR = STAGE_A_DIR / "predictions"
REVIEW_DIR = STAGE_A_DIR / "review"
DEBUG_DIR = STAGE_A_DIR / "debug"

WEAK_LABELS_CSV = LABELS_DIR / "weak_labels.csv"
MANUAL_LABELS_CSV = LABELS_DIR / "manual_labels.csv"
TRAIN_LABELS_CSV = LABELS_DIR / "train_labels.csv"
REVIEW_QUEUE_CSV = REVIEW_DIR / "review_queue.csv"

STAGE_A_MODEL_PT = MODELS_DIR / "stage_a_model.pt"
STAGE_A_METRICS_JSON = MODELS_DIR / "stage_a_metrics.json"
FRAME_PREDICTIONS_CSV = PREDICTIONS_DIR / "frame_predictions.csv"
CLIP_SEGMENTS_CSV = PREDICTIONS_DIR / "clip_segments.csv"

ALL_STAGE_A_DIRS = [
    STAGE_A_DIR,
    FRAMES_DIR,
    EMBEDDINGS_DIR,
    LABELS_DIR,
    MODELS_DIR,
    PREDICTIONS_DIR,
    REVIEW_DIR,
    DEBUG_DIR,
]


def ensure_stage_a_dirs() -> None:
    """Create the Stage A directory tree if it does not already exist."""
    for path in ALL_STAGE_A_DIRS:
        path.mkdir(parents=True, exist_ok=True)
