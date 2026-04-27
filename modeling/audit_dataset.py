#!/usr/bin/env python3
"""
Audit the current Stage B binary-classification dataset.

Usage:
    python -m modeling.audit_dataset
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from stage_b.export_sequences import OFFSPEED_TYPES
from stage_b.paths import FINAL_EVENTS_CSV, SEQUENCES_DIR


LABELS = ("fastball", "offspeed")
SPLITS = ("train", "val", "test")


def binary_label(pitch_type: str) -> str:
    """Map original pitch types to the binary classifier label."""
    return "offspeed" if pitch_type in OFFSPEED_TYPES else "fastball"


def load_final_events(path: Path) -> list[dict]:
    """Load final Stage B event labels."""
    if not path.exists():
        raise FileNotFoundError(f"Missing final events CSV at {path}. Run stage_b.prepare_events first.")

    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        row["release_frame_idx"] = int(row["release_frame_idx"])
        row["catch_frame_idx"] = int(row["catch_frame_idx"])
        row["frame_span"] = row["catch_frame_idx"] - row["release_frame_idx"] + 1
        row["binary_label"] = binary_label(row["pitch_type"])
    return rows


def list_sequence_files(sequences_dir: Path) -> dict[tuple[str, str], list[Path]]:
    """Return sequence files grouped by split and label."""
    files = {}
    for split in SPLITS:
        for label in LABELS:
            directory = sequences_dir / split / label
            files[(split, label)] = sorted(directory.glob("*.npy")) if directory.exists() else []
    return files


def sequence_clip_id(path: Path) -> str:
    """Return clip ID from a sequence file path."""
    return path.stem


def summarize_counter(title: str, counter: Counter) -> None:
    """Print sorted counter values."""
    print(title)
    if not counter:
        print("  none")
        return
    for key, count in sorted(counter.items()):
        print(f"  {key}: {count}")


def span_stats(events: list[dict]) -> dict[str, float | int]:
    """Return release-to-catch span statistics."""
    spans = np.array([event["frame_span"] for event in events], dtype=np.float32)
    if spans.size == 0:
        return {}
    return {
        "min": int(spans.min()),
        "p25": float(np.percentile(spans, 25)),
        "median": float(np.median(spans)),
        "mean": float(spans.mean()),
        "p75": float(np.percentile(spans, 75)),
        "max": int(spans.max()),
    }


def audit_sequence_files(files_by_split_label: dict[tuple[str, str], list[Path]]) -> tuple[dict, list[str]]:
    """Inspect sequence shapes/dtypes and report unreadable files."""
    shape_counts: Counter = Counter()
    dtype_counts: Counter = Counter()
    value_min = None
    value_max = None
    corrupt_files = []

    for files in files_by_split_label.values():
        for path in files:
            try:
                array = np.load(path, mmap_mode="r")
            except Exception as exc:  # noqa: BLE001 - audit should continue through bad files.
                corrupt_files.append(f"{path}: {exc}")
                continue

            shape_counts[str(tuple(array.shape))] += 1
            dtype_counts[str(array.dtype)] += 1
            array_min = float(np.min(array))
            array_max = float(np.max(array))
            value_min = array_min if value_min is None else min(value_min, array_min)
            value_max = array_max if value_max is None else max(value_max, array_max)

    summary = {
        "shape_counts": shape_counts,
        "dtype_counts": dtype_counts,
        "value_min": value_min,
        "value_max": value_max,
    }
    return summary, corrupt_files


def print_split_counts(files_by_split_label: dict[tuple[str, str], list[Path]]) -> None:
    """Print split and label counts from exported sequences."""
    print("Exported sequence counts:")
    for split in SPLITS:
        split_total = sum(len(files_by_split_label[(split, label)]) for label in LABELS)
        print(f"  {split}: {split_total}")
        for label in LABELS:
            print(f"    {label}: {len(files_by_split_label[(split, label)])}")


def compare_events_to_sequences(
    events: list[dict],
    files_by_split_label: dict[tuple[str, str], list[Path]],
) -> tuple[list[str], list[str]]:
    """Return missing and extra sequence clip IDs compared with final events."""
    event_clip_ids = {event["clip_id"] for event in events}
    sequence_clip_ids = {
        sequence_clip_id(path)
        for files in files_by_split_label.values()
        for path in files
    }
    missing = sorted(event_clip_ids - sequence_clip_ids)
    extra = sorted(sequence_clip_ids - event_clip_ids)
    return missing, extra


def print_events_summary(events: list[dict]) -> None:
    """Print label/event distribution summaries."""
    print(f"Final events: {len(events)}")
    summarize_counter("Binary label counts:", Counter(event["binary_label"] for event in events))
    summarize_counter("Pitch type counts:", Counter(event["pitch_type"] for event in events))
    summarize_counter("Event source counts:", Counter(event["event_source"] for event in events))

    spans = span_stats(events)
    print("Release-to-catch span stats:")
    if not spans:
        print("  none")
        return
    for key, value in spans.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")


def print_sequence_audit(summary: dict, corrupt_files: list[str]) -> None:
    """Print sequence array audit results."""
    summarize_counter("Sequence shape counts:", summary["shape_counts"])
    summarize_counter("Sequence dtype counts:", summary["dtype_counts"])
    if summary["value_min"] is not None and summary["value_max"] is not None:
        print("Sequence value range:")
        print(f"  min: {summary['value_min']:.4f}")
        print(f"  max: {summary['value_max']:.4f}")

    print(f"Corrupt/unreadable sequence files: {len(corrupt_files)}")
    for item in corrupt_files[:10]:
        print(f"  {item}")
    if len(corrupt_files) > 10:
        print(f"  ... {len(corrupt_files) - 10} more")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the Stage B sequence dataset")
    parser.add_argument("--events-csv", type=Path, default=FINAL_EVENTS_CSV, help="Final Stage B events CSV")
    parser.add_argument("--sequences-dir", type=Path, default=SEQUENCES_DIR, help="Exported sequence root")
    args = parser.parse_args()

    events = load_final_events(args.events_csv)
    files_by_split_label = list_sequence_files(args.sequences_dir)
    sequence_summary, corrupt_files = audit_sequence_files(files_by_split_label)
    missing_clip_ids, extra_clip_ids = compare_events_to_sequences(events, files_by_split_label)

    print_events_summary(events)
    print()
    print_split_counts(files_by_split_label)
    print()
    print_sequence_audit(sequence_summary, corrupt_files)
    print()
    print(f"Final events missing exported sequences: {len(missing_clip_ids)}")
    for clip_id in missing_clip_ids[:10]:
        print(f"  {clip_id}")
    if len(missing_clip_ids) > 10:
        print(f"  ... {len(missing_clip_ids) - 10} more")

    print(f"Exported sequences without final events: {len(extra_clip_ids)}")
    for clip_id in extra_clip_ids[:10]:
        print(f"  {clip_id}")
    if len(extra_clip_ids) > 10:
        print(f"  ... {len(extra_clip_ids) - 10} more")


if __name__ == "__main__":
    main()
