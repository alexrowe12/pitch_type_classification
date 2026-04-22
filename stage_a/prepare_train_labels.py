#!/usr/bin/env python3
"""
Merge weak and manual Stage A labels into a training CSV.

Usage:
    python -m stage_a.prepare_train_labels
    python -m stage_a.prepare_train_labels --min-weak-confidence 0.8
"""

import argparse
import csv
from pathlib import Path

from stage_a.paths import (
    MANUAL_LABELS_CSV,
    TRAIN_LABELS_CSV,
    WEAK_LABELS_CSV,
    ensure_stage_a_dirs,
)


VALID_LABELS = {"pitch_camera", "non_pitch_camera"}


def load_csv_rows(path: Path) -> list[dict]:
    """Load CSV rows if the file exists, else return an empty list."""
    if not path.exists():
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def build_manual_lookup(rows: list[dict]) -> dict[tuple[str, str], dict]:
    """Build a lookup of manual labels keyed by clip/frame."""
    lookup = {}
    for row in rows:
        assigned_label = row.get("assigned_label", "")
        if assigned_label not in VALID_LABELS:
            continue
        key = (row["clip_id"], row["frame_idx"])
        lookup[key] = row
    return lookup


def merge_labels(
    weak_rows: list[dict],
    manual_rows: list[dict],
    min_weak_confidence: float,
) -> list[dict]:
    """Merge weak and manual labels into a clean training dataset."""
    manual_lookup = build_manual_lookup(manual_rows)
    merged = []

    for row in weak_rows:
        key = (row["clip_id"], row["frame_idx"])
        manual_row = manual_lookup.get(key)
        if manual_row is not None:
            merged.append(
                {
                    "clip_id": row["clip_id"],
                    "frame_idx": row["frame_idx"],
                    "frame_path": row["frame_path"],
                    "pitch_type": row["pitch_type"],
                    "label": manual_row["assigned_label"],
                    "label_source": "manual",
                    "label_confidence": "1.0000",
                }
            )
            continue

        weak_label = row.get("weak_label", "")
        weak_confidence = float(row.get("weak_confidence", 0.0) or 0.0)
        if weak_label not in VALID_LABELS:
            continue
        if weak_confidence < min_weak_confidence:
            continue

        merged.append(
            {
                "clip_id": row["clip_id"],
                "frame_idx": row["frame_idx"],
                "frame_path": row["frame_path"],
                "pitch_type": row["pitch_type"],
                "label": weak_label,
                "label_source": row.get("weak_source", "weak"),
                "label_confidence": f"{weak_confidence:.4f}",
            }
        )

    return merged


def write_train_labels(rows: list[dict]) -> None:
    """Write the merged training labels CSV."""
    TRAIN_LABELS_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "clip_id",
        "frame_idx",
        "frame_path",
        "pitch_type",
        "label",
        "label_source",
        "label_confidence",
    ]
    with open(TRAIN_LABELS_CSV, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> None:
    """Print a compact summary of merged labels."""
    label_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for row in rows:
        label_counts[row["label"]] = label_counts.get(row["label"], 0) + 1
        source_counts[row["label_source"]] = source_counts.get(row["label_source"], 0) + 1

    print("Merged label summary:")
    for label in sorted(label_counts):
        print(f"  label={label}: {label_counts[label]}")
    for source in sorted(source_counts):
        print(f"  source={source}: {source_counts[source]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare merged Stage A training labels")
    parser.add_argument(
        "--min-weak-confidence",
        type=float,
        default=0.80,
        help="Minimum weak-label confidence to include in training",
    )
    args = parser.parse_args()

    ensure_stage_a_dirs()
    weak_rows = load_csv_rows(WEAK_LABELS_CSV)
    if not weak_rows:
        raise FileNotFoundError(
            f"No weak labels found at {WEAK_LABELS_CSV}. Run stage_a.build_weak_labels first."
        )
    manual_rows = load_csv_rows(MANUAL_LABELS_CSV)

    merged_rows = merge_labels(
        weak_rows=weak_rows,
        manual_rows=manual_rows,
        min_weak_confidence=args.min_weak_confidence,
    )
    write_train_labels(merged_rows)

    print(f"Wrote {len(merged_rows)} training row(s) to: {TRAIN_LABELS_CSV}")
    summarize(merged_rows)


if __name__ == "__main__":
    main()
