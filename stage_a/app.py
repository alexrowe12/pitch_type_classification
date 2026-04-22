#!/usr/bin/env python3
"""
Local Streamlit app for Stage A manual labeling.

Usage:
    streamlit run stage_a/app.py
"""

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stage_a.paths import MANUAL_LABELS_CSV, REVIEW_QUEUE_CSV, ensure_stage_a_dirs


def load_csv_rows(path: Path) -> list[dict]:
    """Load CSV rows if present, else return an empty list."""
    if not path.exists():
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def load_manual_labels() -> list[dict]:
    """Load any existing manual labels."""
    return load_csv_rows(MANUAL_LABELS_CSV)


def load_review_queue() -> list[dict]:
    """Load the review queue."""
    return load_csv_rows(REVIEW_QUEUE_CSV)


def save_manual_label(row: dict, assigned_label: str) -> None:
    """Append one manual label row to the manual-label CSV."""
    MANUAL_LABELS_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "clip_id",
        "frame_idx",
        "frame_path",
        "pitch_type",
        "weak_label",
        "weak_confidence",
        "queue_reason",
        "assigned_label",
        "labeled_at_utc",
    ]

    file_exists = MANUAL_LABELS_CSV.exists()
    with open(MANUAL_LABELS_CSV, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "clip_id": row["clip_id"],
                "frame_idx": row["frame_idx"],
                "frame_path": row["frame_path"],
                "pitch_type": row["pitch_type"],
                "weak_label": row["weak_label"],
                "weak_confidence": row["weak_confidence"],
                "queue_reason": row["queue_reason"],
                "assigned_label": assigned_label,
                "labeled_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )


def build_context_paths(frame_path: Path) -> list[Path]:
    """Return nearby frame paths from the same clip directory for context."""
    clip_dir = frame_path.parent
    current_name = frame_path.name
    frame_files = sorted(clip_dir.glob("frame_*.jpg"))
    if not frame_files:
        return [frame_path]

    try:
        idx = frame_files.index(frame_path)
    except ValueError:
        return [frame_path]

    context = []
    for neighbor in [idx - 1, idx, idx + 1]:
        if 0 <= neighbor < len(frame_files):
            context.append(frame_files[neighbor])
    return context


def get_next_item(queue_rows: list[dict], manual_rows: list[dict]) -> dict | None:
    """Return the next unlabeled queue item."""
    labeled_ids = {(row["clip_id"], row["frame_idx"]) for row in manual_rows}
    for row in queue_rows:
        if (row["clip_id"], row["frame_idx"]) not in labeled_ids:
            return row
    return None


def main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:
        raise SystemExit(
            "Streamlit is not installed. Install it with `pip install streamlit`."
        ) from exc

    ensure_stage_a_dirs()
    st.set_page_config(page_title="Stage A Label Review", layout="wide")
    st.title("Stage A Review")

    queue_rows = load_review_queue()
    manual_rows = load_manual_labels()

    if not queue_rows:
        st.warning("No review queue found. Run `python -m stage_a.make_review_queue` first.")
        return

    next_item = get_next_item(queue_rows, manual_rows)
    labeled_count = len(manual_rows)
    st.caption(f"Labeled {labeled_count} of {len(queue_rows)} queued frames")

    if next_item is None:
        st.success("Review queue complete.")
        return

    frame_path = Path(next_item["frame_path"])
    context_paths = build_context_paths(frame_path)

    st.subheader(f"{next_item['clip_id']} frame {next_item['frame_idx']}")
    meta_cols = st.columns(4)
    meta_cols[0].metric("Pitch Type", next_item["pitch_type"])
    meta_cols[1].metric("Weak Label", next_item["weak_label"])
    meta_cols[2].metric("Weak Confidence", next_item["weak_confidence"])
    meta_cols[3].metric("Queue Reason", next_item["queue_reason"])

    image_cols = st.columns(len(context_paths))
    for col, path in zip(image_cols, context_paths):
        col.image(str(path), caption=path.name, use_container_width=True)

    action_cols = st.columns(3)
    if action_cols[0].button("Pitch Camera", use_container_width=True):
        save_manual_label(next_item, "pitch_camera")
        st.rerun()
    if action_cols[1].button("Non Pitch Camera", use_container_width=True):
        save_manual_label(next_item, "non_pitch_camera")
        st.rerun()
    if action_cols[2].button("Skip", use_container_width=True):
        save_manual_label(next_item, "skip")
        st.rerun()


if __name__ == "__main__":
    main()
