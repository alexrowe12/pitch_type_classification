#!/usr/bin/env python3
"""
Export final Stage B pitch sequences into train/val/test numpy arrays.

Usage:
    python -m stage_b.export_sequences --num-frames 12
    python -m stage_b.export_sequences --num-frames 8 --limit 50
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from preprocess.paths import CLIPS_DIR
from stage_b.paths import FINAL_EVENTS_CSV, SEQUENCES_DIR, ensure_stage_b_dirs


DEFAULT_NUM_FRAMES = 12
DEFAULT_IMAGE_SIZE = 224
DEFAULT_CROP_LEFT = 0.15
DEFAULT_CROP_TOP = 0.20
DEFAULT_CROP_RIGHT = 0.90
DEFAULT_CROP_BOTTOM = 0.92
OFFSPEED_TYPES = {"slider", "curveball", "changeup", "sinker", "knucklecurve"}
SUBSET_TO_SPLIT = {"training": "train", "validation": "val", "testing": "test"}


def load_final_events(path: Path) -> list[dict]:
    """Load final Stage B events."""
    if not path.exists():
        raise FileNotFoundError(f"Missing final Stage B events at {path}. Run stage_b.prepare_events first.")
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        row["release_frame_idx"] = int(row["release_frame_idx"])
        row["catch_frame_idx"] = int(row["catch_frame_idx"])
    return rows


def load_clip_metadata(path: Path) -> dict[str, dict]:
    """Load clip split metadata from the downloader output."""
    if not path.exists():
        raise FileNotFoundError(f"Missing clip metadata CSV at {path}.")
    metadata = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            metadata[row["clip_id"]] = row
    return metadata


def find_clip_path(clip_id: str, pitch_type: str) -> Path | None:
    """Find the raw clip path."""
    direct_path = CLIPS_DIR / pitch_type / f"{clip_id}.mp4"
    if direct_path.exists():
        return direct_path
    matches = sorted(CLIPS_DIR.rglob(f"{clip_id}.mp4"))
    if matches:
        return matches[0]
    return None


def clamp_crop_bounds(
    width: int,
    height: int,
    left: float,
    top: float,
    right: float,
    bottom: float,
) -> tuple[int, int, int, int]:
    """Convert normalized crop bounds to safe pixel bounds."""
    x1 = max(0, min(width - 1, int(round(width * left))))
    y1 = max(0, min(height - 1, int(round(height * top))))
    x2 = max(x1 + 1, min(width, int(round(width * right))))
    y2 = max(y1 + 1, min(height, int(round(height * bottom))))
    return x1, y1, x2, y2


def crop_and_resize(
    frame_bgr,
    crop_bounds: tuple[float, float, float, float],
    image_size: int,
):
    """Crop to the pitcher-to-plate action zone and resize."""
    height, width = frame_bgr.shape[:2]
    x1, y1, x2, y2 = clamp_crop_bounds(width, height, *crop_bounds)
    cropped = frame_bgr[y1:y2, x1:x2]
    return cv2.resize(cropped, (image_size, image_size), interpolation=cv2.INTER_AREA)


def choose_frame_indices(release_frame_idx: int, catch_frame_idx: int, num_frames: int) -> list[int]:
    """Choose evenly spaced frame indices between release and catch, inclusive."""
    if catch_frame_idx <= release_frame_idx:
        return [release_frame_idx] * num_frames
    indices = np.linspace(release_frame_idx, catch_frame_idx, num=num_frames)
    return [int(round(value)) for value in indices]


def extract_sequence(
    clip_path: Path,
    frame_indices: list[int],
    crop_bounds: tuple[float, float, float, float],
    image_size: int,
) -> np.ndarray | None:
    """Extract a sequence from the raw clip and return normalized RGB frames."""
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return None

    target_indices = set(frame_indices)
    start_frame = min(frame_indices)
    end_frame = max(frame_indices)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames_by_index = {}
    for frame_idx in range(start_frame, end_frame + 1):
        success, frame_bgr = cap.read()
        if not success:
            break
        if frame_idx not in target_indices:
            continue
        output_frame = crop_and_resize(frame_bgr, crop_bounds=crop_bounds, image_size=image_size)
        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        frames_by_index[frame_idx] = output_frame

    cap.release()

    if any(frame_idx not in frames_by_index for frame_idx in frame_indices):
        return None
    return np.stack([frames_by_index[frame_idx] for frame_idx in frame_indices], axis=0)


def binary_label(pitch_type: str) -> str:
    """Map the original pitch type to fastball vs offspeed."""
    return "offspeed" if pitch_type in OFFSPEED_TYPES else "fastball"


def output_path(split: str, label: str, clip_id: str) -> Path:
    """Return the sequence output path."""
    return SEQUENCES_DIR / split / label / f"{clip_id}.npy"


def export_event_sequence(
    event: dict,
    metadata_by_clip: dict[str, dict],
    num_frames: int,
    image_size: int,
    crop_bounds: tuple[float, float, float, float],
    overwrite: bool,
) -> tuple[str | None, str | None]:
    """Export one event sequence and return (saved_path, error_reason)."""
    clip_id = event["clip_id"]
    pitch_type = event["pitch_type"]
    clip_path = find_clip_path(clip_id, pitch_type)
    if clip_path is None:
        return None, "missing_clip"

    metadata_row = metadata_by_clip.get(clip_id)
    if metadata_row is None:
        return None, "missing_metadata"

    split = SUBSET_TO_SPLIT.get(metadata_row["subset"])
    if split is None:
        return None, "unknown_subset"

    label = binary_label(pitch_type)
    save_path = output_path(split=split, label=label, clip_id=clip_id)
    if save_path.exists() and not overwrite:
        return str(save_path), None

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
        return None, "extract_failed"

    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, sequence)
    return str(save_path), None


def summarize(saved_rows: list[dict], skipped: dict[str, int]) -> None:
    """Print export summary."""
    split_counts: dict[str, int] = {}
    label_counts: dict[str, int] = {}
    for row in saved_rows:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
        label_counts[row["label"]] = label_counts.get(row["label"], 0) + 1

    print("Sequence export summary:")
    for split, count in sorted(split_counts.items()):
        print(f"  split={split}: {count}")
    for label, count in sorted(label_counts.items()):
        print(f"  label={label}: {count}")
    if skipped:
        print("Skipped clips:")
        for reason, count in sorted(skipped.items()):
            print(f"  {reason}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export final Stage B pitch sequences")
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES, help="Frames per exported sequence")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE, help="Output frame size")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of clips to export")
    parser.add_argument("--crop-left", type=float, default=DEFAULT_CROP_LEFT, help="Normalized left crop bound")
    parser.add_argument("--crop-top", type=float, default=DEFAULT_CROP_TOP, help="Normalized top crop bound")
    parser.add_argument("--crop-right", type=float, default=DEFAULT_CROP_RIGHT, help="Normalized right crop bound")
    parser.add_argument("--crop-bottom", type=float, default=DEFAULT_CROP_BOTTOM, help="Normalized bottom crop bound")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing exported sequences")
    args = parser.parse_args()

    ensure_stage_b_dirs()
    crop_bounds = (args.crop_left, args.crop_top, args.crop_right, args.crop_bottom)
    events = load_final_events(FINAL_EVENTS_CSV)
    if args.limit is not None:
        events = events[: args.limit]

    metadata_by_clip = load_clip_metadata(CLIPS_DIR / "metadata.csv")
    print(f"Found {len(events)} final event(s) to export")
    print(f"Using num_frames={args.num_frames}, image_size={args.image_size}, crop={crop_bounds}")

    saved_rows = []
    skipped: dict[str, int] = {}
    for event in tqdm(events, desc="Exporting Stage B sequences"):
        save_path, error_reason = export_event_sequence(
            event=event,
            metadata_by_clip=metadata_by_clip,
            num_frames=args.num_frames,
            image_size=args.image_size,
            crop_bounds=crop_bounds,
            overwrite=args.overwrite,
        )
        if error_reason is not None:
            skipped[error_reason] = skipped.get(error_reason, 0) + 1
            continue

        metadata_row = metadata_by_clip[event["clip_id"]]
        saved_rows.append(
            {
                "clip_id": event["clip_id"],
                "split": SUBSET_TO_SPLIT[metadata_row["subset"]],
                "label": binary_label(event["pitch_type"]),
                "path": save_path,
            }
        )

    print(f"Exported {len(saved_rows)} sequence file(s) under: {SEQUENCES_DIR}")
    summarize(saved_rows, skipped)


if __name__ == "__main__":
    main()
