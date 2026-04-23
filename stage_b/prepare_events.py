#!/usr/bin/env python3
"""
Prepare final Stage B release/catch events from manual review and weak guesses.

Usage:
    python -m stage_b.prepare_events
    python -m stage_b.prepare_events --include-weak --min-weak-confidence 0.85
"""

import argparse
import csv
from pathlib import Path

from stage_b.paths import FINAL_EVENTS_CSV, MANUAL_EVENTS_CSV, WEAK_EVENTS_CSV, ensure_stage_b_dirs


def load_csv_rows(path: Path) -> list[dict]:
    """Load CSV rows if the file exists, else return an empty list."""
    if not path.exists():
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def build_manual_lookup(rows: list[dict]) -> dict[str, dict]:
    """Build manual-event lookup keyed by clip id."""
    lookup = {}
    for row in rows:
        row["release_frame_idx"] = int(row["release_frame_idx"])
        row["catch_frame_idx"] = int(row["catch_frame_idx"])
        row["usable"] = row["usable"] == "1"
        lookup[row["clip_id"]] = row
    return lookup


def merge_events(
    weak_rows: list[dict],
    manual_rows: list[dict],
    include_weak: bool,
    min_weak_confidence: float,
) -> list[dict]:
    """Merge manual and weak Stage B events into final usable event rows."""
    manual_lookup = build_manual_lookup(manual_rows)
    final_rows = []

    for row in weak_rows:
        clip_id = row["clip_id"]
        weak_release = int(row["release_frame_idx"])
        weak_catch = int(row["catch_frame_idx"])
        weak_confidence = float(row.get("confidence", 0.0) or 0.0)
        manual_row = manual_lookup.get(clip_id)

        if manual_row is not None:
            if not manual_row["usable"]:
                continue
            final_rows.append(
                {
                    "clip_id": clip_id,
                    "pitch_type": row["pitch_type"],
                    "release_frame_idx": manual_row["release_frame_idx"],
                    "catch_frame_idx": manual_row["catch_frame_idx"],
                    "event_source": "manual",
                    "event_confidence": "1.000000",
                    "weak_release_frame_idx": weak_release,
                    "weak_catch_frame_idx": weak_catch,
                    "weak_confidence": f"{weak_confidence:.6f}",
                    "notes": manual_row.get("notes", ""),
                }
            )
            continue

        if not include_weak or weak_confidence < min_weak_confidence:
            continue

        final_rows.append(
            {
                "clip_id": clip_id,
                "pitch_type": row["pitch_type"],
                "release_frame_idx": weak_release,
                "catch_frame_idx": weak_catch,
                "event_source": "weak",
                "event_confidence": f"{weak_confidence:.6f}",
                "weak_release_frame_idx": weak_release,
                "weak_catch_frame_idx": weak_catch,
                "weak_confidence": f"{weak_confidence:.6f}",
                "notes": "",
            }
        )

    final_rows.sort(key=lambda row: row["clip_id"])
    return final_rows


def write_final_events(rows: list[dict]) -> None:
    """Write merged final Stage B events."""
    FINAL_EVENTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "clip_id",
        "pitch_type",
        "release_frame_idx",
        "catch_frame_idx",
        "event_source",
        "event_confidence",
        "weak_release_frame_idx",
        "weak_catch_frame_idx",
        "weak_confidence",
        "notes",
    ]
    with open(FINAL_EVENTS_CSV, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> None:
    """Print final-event summary."""
    source_counts: dict[str, int] = {}
    pitch_type_counts: dict[str, int] = {}
    for row in rows:
        source_counts[row["event_source"]] = source_counts.get(row["event_source"], 0) + 1
        pitch_type_counts[row["pitch_type"]] = pitch_type_counts.get(row["pitch_type"], 0) + 1

    print("Final-event summary:")
    for source, count in sorted(source_counts.items()):
        print(f"  source={source}: {count}")
    for pitch_type, count in sorted(pitch_type_counts.items()):
        print(f"  pitch_type={pitch_type}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare final Stage B release/catch events")
    parser.add_argument(
        "--include-weak",
        action="store_true",
        help="Include high-confidence weak events for clips without manual review",
    )
    parser.add_argument(
        "--min-weak-confidence",
        type=float,
        default=0.85,
        help="Minimum weak-event confidence when --include-weak is enabled",
    )
    args = parser.parse_args()

    ensure_stage_b_dirs()
    weak_rows = load_csv_rows(WEAK_EVENTS_CSV)
    if not weak_rows:
        raise FileNotFoundError(
            f"No weak events found at {WEAK_EVENTS_CSV}. Run stage_b.build_weak_events first."
        )
    manual_rows = load_csv_rows(MANUAL_EVENTS_CSV)

    final_rows = merge_events(
        weak_rows=weak_rows,
        manual_rows=manual_rows,
        include_weak=args.include_weak,
        min_weak_confidence=args.min_weak_confidence,
    )
    write_final_events(final_rows)

    print(f"Wrote {len(final_rows)} final event row(s) to: {FINAL_EVENTS_CSV}")
    summarize(final_rows)


if __name__ == "__main__":
    main()
