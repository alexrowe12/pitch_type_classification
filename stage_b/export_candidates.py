#!/usr/bin/env python3
"""
Export candidate Stage B frames from strong Stage A pitch-camera segments.

Usage:
    python -m stage_b.export_candidates
    python -m stage_b.export_candidates --limit 20 --overwrite
"""

import argparse
import csv
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from tqdm import tqdm

from preprocess.paths import CLIPS_DIR
from stage_a.paths import CLIP_SEGMENTS_CSV
from stage_b.paths import FRAME_EXPORTS_CSV, FRAMES_DIR, ensure_stage_b_dirs


DEFAULT_MIN_STAGE_A_PROB = 0.98
DEFAULT_MIN_SEGMENT_FRAMES = 5
DEFAULT_STRIDE = 1
DEFAULT_IMAGE_WIDTH = 224
DEFAULT_IMAGE_HEIGHT = 224
DEFAULT_JPEG_QUALITY = 92
DEFAULT_CROP_LEFT = 0.15
DEFAULT_CROP_TOP = 0.20
DEFAULT_CROP_RIGHT = 0.90
DEFAULT_CROP_BOTTOM = 0.92
DEFAULT_WORKERS = min(8, os.cpu_count() or 1)


def load_stage_a_segments(path: Path) -> list[dict]:
    """Load Stage A clip-level pitch-camera segments."""
    if not path.exists():
        raise FileNotFoundError(f"Missing Stage A segments at {path}. Run stage_a.infer_stage_a first.")
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        row["segment_start_frame"] = int(row["segment_start_frame"])
        row["segment_end_frame"] = int(row["segment_end_frame"])
        row["segment_mean_probability"] = float(row["segment_mean_probability"])
        row["segment_num_sampled_frames"] = int(row["segment_num_sampled_frames"])
    return rows


def filter_segments(
    rows: list[dict],
    min_stage_a_prob: float,
    min_segment_frames: int,
) -> list[dict]:
    """Keep only strong Stage A segments for Stage B."""
    filtered = [
        row
        for row in rows
        if row["segment_mean_probability"] >= min_stage_a_prob
        and row["segment_num_sampled_frames"] >= min_segment_frames
        and row["segment_end_frame"] >= row["segment_start_frame"]
    ]
    return sorted(filtered, key=lambda row: row["clip_id"])


def find_clip_path(clip_id: str, pitch_type: str) -> Path | None:
    """Find the raw video path for a clip."""
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
    image_width: int,
    image_height: int,
):
    """Crop to the pitcher-to-plate action zone and resize."""
    height, width = frame_bgr.shape[:2]
    x1, y1, x2, y2 = clamp_crop_bounds(width, height, *crop_bounds)
    cropped = frame_bgr[y1:y2, x1:x2]
    return cv2.resize(cropped, (image_width, image_height), interpolation=cv2.INTER_AREA)


def collect_frame_indices(start_frame: int, end_frame: int, stride: int) -> list[int]:
    """Collect frame indices from a Stage A segment."""
    stride = max(1, stride)
    indices = list(range(start_frame, end_frame + 1, stride))
    if indices and indices[-1] != end_frame:
        indices.append(end_frame)
    return indices


def export_segment_frames(
    segment: dict,
    stride: int,
    crop_bounds: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    jpeg_quality: int,
    overwrite: bool,
) -> tuple[list[dict], str | None]:
    """Export cropped candidate frames for one Stage A segment."""
    clip_id = segment["clip_id"]
    pitch_type = segment["pitch_type"]
    clip_path = find_clip_path(clip_id, pitch_type)
    if clip_path is None:
        return [], "missing_clip"

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return [], "open_failed"

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    start_frame = max(0, segment["segment_start_frame"])
    end_frame = min(max(0, total_frames - 1), segment["segment_end_frame"])
    if end_frame < start_frame:
        cap.release()
        return [], "empty_segment"

    clip_output_dir = FRAMES_DIR / clip_id
    clip_output_dir.mkdir(parents=True, exist_ok=True)

    target_indices = set(collect_frame_indices(start_frame, end_frame, stride))
    rows = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for frame_idx in range(start_frame, end_frame + 1):
        success, frame_bgr = cap.read()
        if not success:
            break
        if frame_idx not in target_indices:
            continue

        frame_path = clip_output_dir / f"frame_{frame_idx:04d}.jpg"
        if overwrite or not frame_path.exists():
            output_frame = crop_and_resize(
                frame_bgr=frame_bgr,
                crop_bounds=crop_bounds,
                image_width=image_width,
                image_height=image_height,
            )
            cv2.imwrite(
                str(frame_path),
                output_frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
            )

        rows.append(
            {
                "clip_id": clip_id,
                "pitch_type": pitch_type,
                "source_video": str(clip_path),
                "frame_idx": frame_idx,
                "fps": f"{fps:.6f}",
                "timestamp_sec": f"{(frame_idx / fps) if fps > 0 else 0.0:.6f}",
                "frame_path": str(frame_path),
                "segment_start_frame": start_frame,
                "segment_end_frame": end_frame,
                "segment_mean_probability": f"{segment['segment_mean_probability']:.6f}",
                "segment_num_sampled_frames": segment["segment_num_sampled_frames"],
                "crop_left": f"{crop_bounds[0]:.4f}",
                "crop_top": f"{crop_bounds[1]:.4f}",
                "crop_right": f"{crop_bounds[2]:.4f}",
                "crop_bottom": f"{crop_bounds[3]:.4f}",
                "image_width": image_width,
                "image_height": image_height,
            }
        )

    cap.release()
    return rows, None


def write_frame_exports(rows: list[dict]) -> None:
    """Write Stage B candidate frame export metadata."""
    FRAME_EXPORTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "clip_id",
        "pitch_type",
        "source_video",
        "frame_idx",
        "fps",
        "timestamp_sec",
        "frame_path",
        "segment_start_frame",
        "segment_end_frame",
        "segment_mean_probability",
        "segment_num_sampled_frames",
        "crop_left",
        "crop_top",
        "crop_right",
        "crop_bottom",
        "image_width",
        "image_height",
    ]
    with open(FRAME_EXPORTS_CSV, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Stage B candidate frames")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of clips to export")
    parser.add_argument(
        "--min-stage-a-prob",
        type=float,
        default=DEFAULT_MIN_STAGE_A_PROB,
        help=f"Minimum Stage A segment mean probability (default: {DEFAULT_MIN_STAGE_A_PROB})",
    )
    parser.add_argument(
        "--min-segment-frames",
        type=int,
        default=DEFAULT_MIN_SEGMENT_FRAMES,
        help=f"Minimum number of sampled Stage A frames in a segment (default: {DEFAULT_MIN_SEGMENT_FRAMES})",
    )
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE, help="Export every Nth frame")
    parser.add_argument("--image-width", type=int, default=DEFAULT_IMAGE_WIDTH, help="Output image width")
    parser.add_argument("--image-height", type=int, default=DEFAULT_IMAGE_HEIGHT, help="Output image height")
    parser.add_argument("--crop-left", type=float, default=DEFAULT_CROP_LEFT, help="Normalized left crop bound")
    parser.add_argument("--crop-top", type=float, default=DEFAULT_CROP_TOP, help="Normalized top crop bound")
    parser.add_argument("--crop-right", type=float, default=DEFAULT_CROP_RIGHT, help="Normalized right crop bound")
    parser.add_argument("--crop-bottom", type=float, default=DEFAULT_CROP_BOTTOM, help="Normalized bottom crop bound")
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=DEFAULT_JPEG_QUALITY,
        help=f"JPEG quality for exported frames (default: {DEFAULT_JPEG_QUALITY})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of clip export workers (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing candidate frames")
    args = parser.parse_args()

    ensure_stage_b_dirs()
    crop_bounds = (args.crop_left, args.crop_top, args.crop_right, args.crop_bottom)

    segments = filter_segments(
        rows=load_stage_a_segments(CLIP_SEGMENTS_CSV),
        min_stage_a_prob=args.min_stage_a_prob,
        min_segment_frames=args.min_segment_frames,
    )
    if args.limit is not None:
        segments = segments[: args.limit]

    print(f"Found {len(segments)} strong Stage A segment(s) to export")
    print(f"Using stride={args.stride}, crop={crop_bounds}, size={args.image_width}x{args.image_height}")
    print(f"Using workers={args.workers}")

    all_rows = []
    skipped: dict[str, int] = {}

    def export_one(segment: dict) -> tuple[list[dict], str | None]:
        return export_segment_frames(
            segment=segment,
            stride=args.stride,
            crop_bounds=crop_bounds,
            image_width=args.image_width,
            image_height=args.image_height,
            jpeg_quality=args.jpeg_quality,
            overwrite=args.overwrite,
        )

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for rows, skip_reason in tqdm(
            executor.map(export_one, segments),
            total=len(segments),
            desc="Exporting Stage B candidates",
        ):
            if skip_reason is not None:
                skipped[skip_reason] = skipped.get(skip_reason, 0) + 1
                continue
            all_rows.extend(rows)

    write_frame_exports(all_rows)
    print(f"Exported {len(all_rows)} candidate frame(s)")
    print(f"Metadata written to: {FRAME_EXPORTS_CSV}")
    print(f"Frames written under: {FRAMES_DIR}")
    if skipped:
        print("Skipped clips:")
        for reason, count in sorted(skipped.items()):
            print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
