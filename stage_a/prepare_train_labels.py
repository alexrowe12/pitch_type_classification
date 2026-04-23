#!/usr/bin/env python3
"""
Merge weak and manual Stage A labels into a training CSV.

Usage:
    python -m stage_a.prepare_train_labels
    python -m stage_a.prepare_train_labels --min-weak-confidence 0.8
"""

import argparse
import csv
import random
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
    max_positive_to_negative_ratio: float,
    random_seed: int,
) -> list[dict]:
    """Merge weak and manual labels into a clean, roughly balanced training dataset."""
    manual_lookup = build_manual_lookup(manual_rows)
    manual_merged = []
    weak_non_pitch_rows = []
    weak_pitch_rows = []

    for row in weak_rows:
        key = (row["clip_id"], row["frame_idx"])
        manual_row = manual_lookup.get(key)
        if manual_row is not None:
            manual_merged.append(
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

        merged_row = {
            "clip_id": row["clip_id"],
            "frame_idx": row["frame_idx"],
            "frame_path": row["frame_path"],
            "pitch_type": row["pitch_type"],
            "label": weak_label,
            "label_source": row.get("weak_source", "weak"),
            "label_confidence": f"{weak_confidence:.4f}",
        }
        if weak_label == "non_pitch_camera":
            weak_non_pitch_rows.append(merged_row)
        else:
            weak_pitch_rows.append(merged_row)

    negative_count = sum(1 for row in manual_merged if row["label"] == "non_pitch_camera")
    negative_count += len(weak_non_pitch_rows)
    manual_pitch_count = sum(1 for row in manual_merged if row["label"] == "pitch_camera")

    if negative_count > 0 and max_positive_to_negative_ratio > 0:
        max_total_pitch = int(negative_count * max_positive_to_negative_ratio)
        weak_pitch_limit = max(0, max_total_pitch - manual_pitch_count)
        rng = random.Random(random_seed)
        if len(weak_pitch_rows) > weak_pitch_limit:
            weak_pitch_rows = rng.sample(weak_pitch_rows, weak_pitch_limit)

    merged = manual_merged + weak_non_pitch_rows + weak_pitch_rows
    rng = random.Random(random_seed)
    rng.shuffle(merged)
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
    negative_count = label_counts.get("non_pitch_camera", 0)
    positive_count = label_counts.get("pitch_camera", 0)
    if negative_count:
        print(f"  pitch_to_non_pitch_ratio: {positive_count / negative_count:.2f}:1")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare merged Stage A training labels")
    parser.add_argument(
        "--min-weak-confidence",
        type=float,
        default=0.80,
        help="Minimum weak-label confidence to include in training",
    )
    parser.add_argument(
        "--max-positive-to-negative-ratio",
        type=float,
        default=4.0,
        help="Maximum pitch_camera:non_pitch_camera ratio after merging labels",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed used when downsampling weak pitch-camera rows",
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
        max_positive_to_negative_ratio=args.max_positive_to_negative_ratio,
        random_seed=args.random_seed,
    )
    write_train_labels(merged_rows)

    print(f"Wrote {len(merged_rows)} training row(s) to: {TRAIN_LABELS_CSV}")
    summarize(merged_rows)


if __name__ == "__main__":
    main()
