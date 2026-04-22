#!/usr/bin/env python3
"""
Build a Stage A manual-review queue from weak labels.

Usage:
    python -m stage_a.make_review_queue
    python -m stage_a.make_review_queue --target-size 300
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from stage_a.paths import (
    MANUAL_LABELS_CSV,
    REVIEW_QUEUE_CSV,
    WEAK_LABELS_CSV,
    ensure_stage_a_dirs,
)


def load_csv_rows(path: Path) -> list[dict]:
    """Load CSV rows if the file exists, else return an empty list."""
    if not path.exists():
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def labeled_frame_ids() -> set[tuple[str, str]]:
    """Return (clip_id, frame_idx) keys already manually labeled."""
    rows = load_csv_rows(MANUAL_LABELS_CSV)
    return {(row["clip_id"], row["frame_idx"]) for row in rows}


def add_priority_rows(
    selected: list[dict],
    seen: set[tuple[str, str]],
    rows: list[dict],
    reason: str,
    per_clip_cap: int,
    max_count: int,
) -> None:
    """Append rows to the queue while respecting dedupe and clip caps."""
    per_clip_counts: dict[str, int] = defaultdict(int)
    for row in selected:
        per_clip_counts[row["clip_id"]] += 1

    added = 0
    for row in rows:
        key = (row["clip_id"], row["frame_idx"])
        if key in seen:
            continue
        if per_clip_counts[row["clip_id"]] >= per_clip_cap:
            continue

        queue_row = {
            "clip_id": row["clip_id"],
            "frame_idx": row["frame_idx"],
            "frame_path": row["frame_path"],
            "pitch_type": row["pitch_type"],
            "weak_label": row["weak_label"],
            "weak_confidence": row["weak_confidence"],
            "queue_reason": reason,
            "queue_priority": str(len(selected) + 1),
        }
        selected.append(queue_row)
        seen.add(key)
        per_clip_counts[row["clip_id"]] += 1
        added += 1

        if added >= max_count:
            break


def build_review_queue(
    weak_rows: list[dict],
    target_size: int,
    per_clip_cap: int,
    unknown_ratio: float,
    audit_ratio: float,
) -> list[dict]:
    """Construct a prioritized review queue from weak labels."""
    already_labeled = labeled_frame_ids()
    selected: list[dict] = []
    seen = set(already_labeled)

    unknown_rows = [
        row for row in weak_rows
        if row["weak_label"] == "unknown" and (row["clip_id"], row["frame_idx"]) not in seen
    ]
    low_conf_rows = sorted(
        [
            row for row in weak_rows
            if row["weak_label"] != "unknown" and (row["clip_id"], row["frame_idx"]) not in seen
        ],
        key=lambda row: float(row["weak_confidence"]),
    )
    pitch_audit_rows = [
        row for row in weak_rows
        if row["weak_label"] == "pitch_camera" and (row["clip_id"], row["frame_idx"]) not in seen
    ]
    non_pitch_audit_rows = [
        row for row in weak_rows
        if row["weak_label"] == "non_pitch_camera" and (row["clip_id"], row["frame_idx"]) not in seen
    ]

    max_unknown = int(target_size * unknown_ratio)
    max_audit = int(target_size * audit_ratio)
    max_low_conf = max(0, target_size - max_unknown - max_audit)

    add_priority_rows(
        selected=selected,
        seen=seen,
        rows=unknown_rows,
        reason="unknown_rule_output",
        per_clip_cap=per_clip_cap,
        max_count=max_unknown,
    )
    add_priority_rows(
        selected=selected,
        seen=seen,
        rows=low_conf_rows,
        reason="low_confidence_weak_label",
        per_clip_cap=per_clip_cap,
        max_count=max_low_conf,
    )

    half_audit = max_audit // 2
    add_priority_rows(
        selected=selected,
        seen=seen,
        rows=pitch_audit_rows,
        reason="pitch_camera_audit",
        per_clip_cap=per_clip_cap,
        max_count=half_audit,
    )
    add_priority_rows(
        selected=selected,
        seen=seen,
        rows=non_pitch_audit_rows,
        reason="non_pitch_camera_audit",
        per_clip_cap=per_clip_cap,
        max_count=max_audit - half_audit,
    )

    return selected[:target_size]


def write_review_queue(rows: list[dict]) -> None:
    """Write the review queue CSV."""
    REVIEW_QUEUE_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "clip_id",
        "frame_idx",
        "frame_path",
        "pitch_type",
        "weak_label",
        "weak_confidence",
        "queue_reason",
        "queue_priority",
    ]
    with open(REVIEW_QUEUE_CSV, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Stage A review queue")
    parser.add_argument(
        "--target-size",
        type=int,
        default=250,
        help="Desired number of frames in the review queue",
    )
    parser.add_argument(
        "--per-clip-cap",
        type=int,
        default=12,
        help="Maximum queued frames from any one clip",
    )
    parser.add_argument(
        "--unknown-ratio",
        type=float,
        default=0.6,
        help="Fraction of queue reserved for unknown weak labels",
    )
    parser.add_argument(
        "--audit-ratio",
        type=float,
        default=0.2,
        help="Fraction of queue reserved for random-ish audits of weak labels",
    )
    args = parser.parse_args()

    ensure_stage_a_dirs()
    weak_rows = load_csv_rows(WEAK_LABELS_CSV)
    if not weak_rows:
        raise FileNotFoundError(
            f"No weak labels found at {WEAK_LABELS_CSV}. Run stage_a.build_weak_labels first."
        )

    queue_rows = build_review_queue(
        weak_rows=weak_rows,
        target_size=args.target_size,
        per_clip_cap=args.per_clip_cap,
        unknown_ratio=args.unknown_ratio,
        audit_ratio=args.audit_ratio,
    )
    write_review_queue(queue_rows)

    print(f"Wrote {len(queue_rows)} review item(s) to: {REVIEW_QUEUE_CSV}")


if __name__ == "__main__":
    main()
