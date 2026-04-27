#!/usr/bin/env python3
"""
Export a configurable RGB sequence set from reviewed Stage B events.

This preserves the current train/val/test split by reading the existing
Stage B sequence directories, then regenerates frames from the raw clips.

Usage:
    python -m modeling.export_sequence_set --num-frames 20 --output-root data/modeling/variants_20
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from modeling.audit_dataset import LABELS, SPLITS
from modeling.paths import MODELING_DIR, ensure_modeling_dirs
from stage_b.export_sequences import (
    DEFAULT_CROP_BOTTOM,
    DEFAULT_CROP_LEFT,
    DEFAULT_CROP_RIGHT,
    DEFAULT_CROP_TOP,
    DEFAULT_IMAGE_SIZE,
    binary_label,
    choose_frame_indices,
    extract_sequence,
    find_clip_path,
    load_final_events,
)
from stage_b.paths import FINAL_EVENTS_CSV, SEQUENCES_DIR


DEFAULT_NUM_FRAMES = 20
DEFAULT_WORKERS = min(8, os.cpu_count() or 1)
DEFAULT_OUTPUT_ROOT = MODELING_DIR / "variants_20"


def load_current_split_map(reference_dir: Path) -> dict[str, dict]:
    """Load clip split/label assignments from an existing sequence directory."""
    split_map = {}
    for split in SPLITS:
        for label in LABELS:
            directory = reference_dir / split / label
            if not directory.exists():
                continue
            for path in sorted(directory.glob("*.npy")):
                split_map[path.stem] = {
                    "split": split,
                    "label": label,
                    "reference_path": path,
                }
    return split_map


def output_path(output_root: Path, split: str, label: str, clip_id: str) -> Path:
    """Return output path for an RGB sequence variant."""
    return output_root / "rgb" / split / label / f"{clip_id}.npy"


def export_event(
    event: dict,
    split_map: dict[str, dict],
    output_root: Path,
    num_frames: int,
    image_size: int,
    crop_bounds: tuple[float, float, float, float],
    overwrite: bool,
) -> tuple[dict, str | None, str | None]:
    """Export one event and return (event, saved_path, error_reason)."""
    clip_id = event["clip_id"]
    pitch_type = event["pitch_type"]
    split_row = split_map.get(clip_id)
    if split_row is None:
        return event, None, "missing_reference_split"

    label = binary_label(pitch_type)
    if split_row["label"] != label:
        return event, None, "label_mismatch"

    save_path = output_path(output_root, split_row["split"], label, clip_id)
    if save_path.exists() and not overwrite:
        return event, str(save_path), None

    clip_path = find_clip_path(clip_id, pitch_type)
    if clip_path is None:
        return event, None, "missing_clip"

    frame_indices = choose_frame_indices(
        release_frame_idx=event["release_frame_idx"],
        catch_frame_idx=event["catch_frame_idx"],
        num_frames=num_frames,
    )
    sequence = extract_sequence(
        clip_path=clip_path,
        frame_indices=frame_indices,
        crop_bounds=crop_bounds,
        image_size=image_size,
    )
    if sequence is None:
        return event, None, "extract_failed"

    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, sequence)
    return event, str(save_path), None


def summarize(saved_rows: list[dict], skipped: dict[str, int], output_root: Path) -> None:
    """Print export summary."""
    print(f"Exported {len(saved_rows)} RGB sequence file(s) under: {output_root / 'rgb'}")
    print("Sequence-set summary:")
    for split in SPLITS:
        split_total = sum(1 for row in saved_rows if row["split"] == split)
        print(f"  {split}: {split_total}")
        for label in LABELS:
            count = sum(1 for row in saved_rows if row["split"] == split and row["label"] == label)
            print(f"    {label}: {count}")
    if skipped:
        print("Skipped clips:")
        for reason, count in sorted(skipped.items()):
            print(f"  {reason}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a configurable modeling RGB sequence set")
    parser.add_argument("--events-csv", type=Path, default=FINAL_EVENTS_CSV, help="Final Stage B events CSV")
    parser.add_argument("--reference-dir", type=Path, default=SEQUENCES_DIR, help="Existing split reference sequence root")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Output variant root")
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES, help="Frames per sequence")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE, help="Output frame size")
    parser.add_argument("--crop-left", type=float, default=DEFAULT_CROP_LEFT, help="Normalized left crop bound")
    parser.add_argument("--crop-top", type=float, default=DEFAULT_CROP_TOP, help="Normalized top crop bound")
    parser.add_argument("--crop-right", type=float, default=DEFAULT_CROP_RIGHT, help="Normalized right crop bound")
    parser.add_argument("--crop-bottom", type=float, default=DEFAULT_CROP_BOTTOM, help="Normalized bottom crop bound")
    parser.add_argument("--limit", type=int, default=None, help="Maximum events to export")
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of sequence export workers (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    args = parser.parse_args()

    ensure_modeling_dirs()
    events = load_final_events(args.events_csv)
    if args.limit is not None:
        events = events[: args.limit]

    split_map = load_current_split_map(args.reference_dir)
    crop_bounds = (args.crop_left, args.crop_top, args.crop_right, args.crop_bottom)
    print(f"Found {len(events)} final event(s)")
    print(f"Loaded {len(split_map)} reference split assignment(s) from: {args.reference_dir}")
    print(f"Exporting RGB sequences with num_frames={args.num_frames}, image_size={args.image_size}")
    print(f"Using crop={crop_bounds}")
    print(f"Using workers={args.workers}")

    saved_rows = []
    skipped: dict[str, int] = {}

    def export_one(event: dict) -> tuple[dict, str | None, str | None]:
        return export_event(
            event=event,
            split_map=split_map,
            output_root=args.output_root,
            num_frames=args.num_frames,
            image_size=args.image_size,
            crop_bounds=crop_bounds,
            overwrite=args.overwrite,
        )

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for event, save_path, error_reason in tqdm(
            executor.map(export_one, events),
            total=len(events),
            desc="Exporting modeling sequence set",
        ):
            if error_reason is not None:
                skipped[error_reason] = skipped.get(error_reason, 0) + 1
                continue

            split_row = split_map[event["clip_id"]]
            saved_rows.append(
                {
                    "clip_id": event["clip_id"],
                    "split": split_row["split"],
                    "label": split_row["label"],
                    "path": save_path,
                }
            )

    summarize(saved_rows, skipped, output_root=args.output_root)


if __name__ == "__main__":
    main()
