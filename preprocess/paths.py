from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESEARCH_DIR = PROJECT_ROOT / "research"

CLIPS_DIR = DATA_DIR / "clips"
PROCESSED_DIR = DATA_DIR / "processed"
DEBUG_DIR = DATA_DIR / "debug"

MLB_YOUTUBE_REPO_DIR = RESEARCH_DIR / "mlb-youtube-repo"
SEGMENTED_JSON = MLB_YOUTUBE_REPO_DIR / "data" / "mlb-youtube-segmented.json"

DOWNLOAD_ERROR_LOG = DATA_DIR / "download_errors.log"
