#!/usr/bin/env python3
"""
Export sampled frames from raw clips for Stage A shot classification.

Usage:
    python -m stage_a.export_frames
    python -m stage_a.export_frames --limit 25
    python -m stage_a.export_frames --stride 12 --overwrite
"""

import argparse
import csv
from pathlib import Path

import cv2
from tqdm import tqdm

from preprocess.paths import CLIPS_DIR
from stage_a.paths import FRAMES_DIR, LABELS_DIR, ensure_stage_a_dirs


FRAME_EXPORTS_CSV = LABELS_DIR / "frame_exports.csv"
DEFAULT_STRIDE = 12
DEFAULT_JPEG_QUALITY = 90


def get_clip_files() -> list[Path]:
    """Return all available raw clip files."""
    if not CLIPS_DIR.exists():
        return []
    return sorted(CLIPS_DIR.rglob("*.mp4"))


def collect_frame_indices(total_frames: int, stride: int) -> list[int]:
    """Choose which frame indices to export for one clip."""
    if total_frames <= 0:
        return []
    indices = list(range(0, total_frames, stride))
    last_frame = total_frames - 1
    if indices[-1] != last_frame:
        indices.append(last_frame)
    return indices


def export_clip_frames(
    clip_path: Path,
    stride: int,
    overwrite: bool,
    jpeg_quality: int,
) -> list[dict]:
    """Export sampled frames for one clip and return metadata rows."""
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    clip_id = clip_path.stem
    pitch_type = clip_path.parent.name
    frame_indices = collect_frame_indices(total_frames, stride)

    clip_output_dir = FRAMES_DIR / clip_id
    clip_output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame_bgr = cap.read()
        if not success:
            continue

        frame_path = clip_output_dir / f"frame_{frame_idx:04d}.jpg"
        if overwrite or not frame_path.exists():
            cv2.imwrite(
                str(frame_path),
                frame_bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
            )

        rows.append(
            {
                "clip_id": clip_id,
                "pitch_type": pitch_type,
                "clip_path": str(clip_path),
                "frame_idx": frame_idx,
                "fps": f"{fps:.6f}",
                "timestamp_sec": f"{(frame_idx / fps) if fps > 0 else 0.0:.6f}",
                "frame_path": str(frame_path),
            }
        )

    cap.release()
    return rows


def write_frame_exports(rows: list[dict]) -> None:
    """Write the exported-frame metadata CSV."""
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "clip_id",
        "pitch_type",
        "clip_path",
        "frame_idx",
        "fps",
        "timestamp_sec",
        "frame_path",
    ]

    with open(FRAME_EXPORTS_CSV, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export sampled frames for Stage A")
    parser.add_argument("--limit", type=int, help="Limit the number of clips to export")
    parser.add_argument(
        "--stride",
        type=int,
        default=DEFAULT_STRIDE,
        help=f"Export every Nth frame (default: {DEFAULT_STRIDE})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite any existing exported JPEG frames",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=DEFAULT_JPEG_QUALITY,
        help=f"JPEG quality for exported frames (default: {DEFAULT_JPEG_QUALITY})",
    )
    args = parser.parse_args()

    ensure_stage_a_dirs()

    clips = get_clip_files()
    if args.limit is not None:
        clips = clips[: args.limit]

    print(f"Found {len(clips)} clip(s) to export")
    print(f"Using stride={args.stride}")

    all_rows = []
    for clip_path in tqdm(clips, desc="Exporting Stage A frames"):
        rows = export_clip_frames(
            clip_path=clip_path,
            stride=args.stride,
            overwrite=args.overwrite,
            jpeg_quality=args.jpeg_quality,
        )
        all_rows.extend(rows)

    write_frame_exports(all_rows)
    print(f"Exported {len(all_rows)} frame(s)")
    print(f"Metadata written to: {FRAME_EXPORTS_CSV}")
    print(f"Frames written under: {FRAMES_DIR}")


if __name__ == "__main__":
    main()
