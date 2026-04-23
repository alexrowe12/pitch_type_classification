#!/usr/bin/env python3
"""
Build weak release/catch guesses for Stage B candidate clips.

Usage:
    python -m stage_b.build_weak_events
    python -m stage_b.build_weak_events --limit 50
"""

import argparse
import csv
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from stage_b.paths import FRAME_EXPORTS_CSV, WEAK_EVENTS_CSV, ensure_stage_b_dirs


MIN_RELEASE_TO_CATCH_GAP = 8
MAX_RELEASE_TO_CATCH_GAP = 42
DEFAULT_SMOOTH_WINDOW = 5
DEFAULT_WORKERS = min(8, os.cpu_count() or 1)


def load_frame_exports(path: Path) -> list[dict]:
    """Load Stage B candidate frame rows."""
    if not path.exists():
        raise FileNotFoundError(f"Missing Stage B frame exports at {path}. Run stage_b.export_candidates first.")
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        row["frame_idx"] = int(row["frame_idx"])
        row["fps"] = float(row["fps"])
        row["timestamp_sec"] = float(row["timestamp_sec"])
        row["segment_start_frame"] = int(row["segment_start_frame"])
        row["segment_end_frame"] = int(row["segment_end_frame"])
        row["segment_mean_probability"] = float(row["segment_mean_probability"])
    return rows


def group_rows(rows: list[dict]) -> dict[str, list[dict]]:
    """Group exported candidate rows by clip."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["clip_id"]].append(row)
    for clip_rows in grouped.values():
        clip_rows.sort(key=lambda row: row["frame_idx"])
    return grouped


def load_gray_frames(rows: list[dict]) -> list[np.ndarray]:
    """Load candidate frames as grayscale arrays."""
    frames = []
    for row in rows:
        image = cv2.imread(row["frame_path"], cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        frames.append(image)
    return frames


def smooth_signal(values: np.ndarray, window_size: int) -> np.ndarray:
    """Smooth a 1D signal with edge padding."""
    if len(values) == 0:
        return values
    window_size = max(1, min(window_size, len(values)))
    kernel = np.ones(window_size, dtype=np.float32) / window_size
    pad_left = window_size // 2
    pad_right = window_size - 1 - pad_left
    padded = np.pad(values.astype(np.float32), (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def region_motion(frames: list[np.ndarray], region: tuple[float, float, float, float]) -> np.ndarray:
    """Compute frame-to-frame motion inside a normalized region."""
    if len(frames) < 2:
        return np.array([], dtype=np.float32)

    height, width = frames[0].shape[:2]
    left, top, right, bottom = region
    x1 = max(0, min(width - 1, int(round(width * left))))
    y1 = max(0, min(height - 1, int(round(height * top))))
    x2 = max(x1 + 1, min(width, int(round(width * right))))
    y2 = max(y1 + 1, min(height, int(round(height * bottom))))

    scores = []
    previous = frames[0][y1:y2, x1:x2]
    for frame in frames[1:]:
        current = frame[y1:y2, x1:x2]
        diff = cv2.absdiff(current, previous)
        scores.append(float(np.mean(diff)))
        previous = current
    return np.array(scores, dtype=np.float32)


def normalized(values: np.ndarray) -> np.ndarray:
    """Min-max normalize a signal."""
    if len(values) == 0:
        return values
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if max_value - min_value < 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - min_value) / (max_value - min_value)).astype(np.float32)


def index_to_frame(rows: list[dict], signal_index: int) -> int:
    """Map a frame-difference signal index to the later frame's original index."""
    return rows[min(signal_index + 1, len(rows) - 1)]["frame_idx"]


def choose_release_index(pitcher_signal: np.ndarray, full_signal: np.ndarray) -> int:
    """Choose a weak release index from early/middle pitcher motion."""
    n = len(pitcher_signal)
    if n == 0:
        return 0

    start = max(0, int(n * 0.08))
    end = max(start + 1, int(n * 0.62))
    search = 0.70 * normalized(pitcher_signal) + 0.30 * normalized(full_signal)
    return start + int(np.argmax(search[start:end]))


def choose_catch_index(
    release_index: int,
    plate_signal: np.ndarray,
    full_signal: np.ndarray,
) -> int:
    """Choose a weak catch index after the release guess."""
    n = len(plate_signal)
    if n == 0:
        return release_index

    start = min(n - 1, release_index + MIN_RELEASE_TO_CATCH_GAP)
    end = min(n, release_index + MAX_RELEASE_TO_CATCH_GAP + 1)
    if end <= start:
        return min(n - 1, release_index + MIN_RELEASE_TO_CATCH_GAP)

    search = 0.65 * normalized(plate_signal) + 0.35 * normalized(full_signal)
    return start + int(np.argmax(search[start:end]))


def confidence_from_signals(
    release_index: int,
    catch_index: int,
    release_signal: np.ndarray,
    catch_signal: np.ndarray,
) -> tuple[float, str]:
    """Estimate weak-label confidence from peak strength and plausible timing."""
    if len(release_signal) == 0 or len(catch_signal) == 0:
        return 0.0, "insufficient_frames"

    gap = catch_index - release_index
    if gap < MIN_RELEASE_TO_CATCH_GAP:
        return 0.1, "gap_too_short"
    if gap > MAX_RELEASE_TO_CATCH_GAP:
        return 0.2, "gap_too_long"

    release_norm = normalized(release_signal)
    catch_norm = normalized(catch_signal)
    release_strength = float(release_norm[release_index])
    catch_strength = float(catch_norm[catch_index])
    confidence = 0.5 * release_strength + 0.5 * catch_strength

    if confidence >= 0.75:
        reason = "strong_motion_peaks"
    elif confidence >= 0.45:
        reason = "moderate_motion_peaks"
    else:
        reason = "weak_motion_peaks"
    return confidence, reason


def build_event_for_clip(rows: list[dict], smooth_window: int) -> dict:
    """Build one weak release/catch event row."""
    frames = load_gray_frames(rows)
    if len(frames) != len(rows):
        return {
            "clip_id": rows[0]["clip_id"],
            "pitch_type": rows[0]["pitch_type"],
            "release_frame_idx": rows[0]["frame_idx"],
            "catch_frame_idx": rows[-1]["frame_idx"],
            "release_signal_score": "0.000000",
            "catch_signal_score": "0.000000",
            "confidence": "0.000000",
            "reason": "missing_frames",
        }

    pitcher_motion = smooth_signal(
        region_motion(frames, region=(0.00, 0.10, 0.48, 0.92)),
        smooth_window,
    )
    plate_motion = smooth_signal(
        region_motion(frames, region=(0.45, 0.12, 1.00, 0.94)),
        smooth_window,
    )
    full_motion = smooth_signal(
        region_motion(frames, region=(0.00, 0.00, 1.00, 1.00)),
        smooth_window,
    )

    release_signal = 0.70 * normalized(pitcher_motion) + 0.30 * normalized(full_motion)
    catch_signal = 0.65 * normalized(plate_motion) + 0.35 * normalized(full_motion)

    release_index = choose_release_index(pitcher_motion, full_motion)
    catch_index = choose_catch_index(release_index, plate_motion, full_motion)
    confidence, reason = confidence_from_signals(
        release_index=release_index,
        catch_index=catch_index,
        release_signal=release_signal,
        catch_signal=catch_signal,
    )

    return {
        "clip_id": rows[0]["clip_id"],
        "pitch_type": rows[0]["pitch_type"],
        "release_frame_idx": index_to_frame(rows, release_index),
        "catch_frame_idx": index_to_frame(rows, catch_index),
        "release_signal_score": f"{float(release_signal[release_index]):.6f}",
        "catch_signal_score": f"{float(catch_signal[catch_index]):.6f}",
        "confidence": f"{confidence:.6f}",
        "reason": reason,
    }


def write_weak_events(rows: list[dict]) -> None:
    """Write weak event guesses to CSV."""
    WEAK_EVENTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "clip_id",
        "pitch_type",
        "release_frame_idx",
        "catch_frame_idx",
        "release_signal_score",
        "catch_signal_score",
        "confidence",
        "reason",
    ]
    with open(WEAK_EVENTS_CSV, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> None:
    """Print weak-event summary."""
    reason_counts: dict[str, int] = {}
    for row in rows:
        reason_counts[row["reason"]] = reason_counts.get(row["reason"], 0) + 1

    print("Weak-event summary:")
    for reason, count in sorted(reason_counts.items()):
        print(f"  {reason}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build weak Stage B release/catch events")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of clips to process")
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=DEFAULT_SMOOTH_WINDOW,
        help=f"Motion signal smoothing window (default: {DEFAULT_SMOOTH_WINDOW})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of clip analysis workers (default: {DEFAULT_WORKERS})",
    )
    args = parser.parse_args()

    ensure_stage_b_dirs()
    grouped = group_rows(load_frame_exports(FRAME_EXPORTS_CSV))
    clip_ids = sorted(grouped)
    if args.limit is not None:
        clip_ids = clip_ids[: args.limit]

    print(f"Using workers={args.workers}")

    def build_one(clip_id: str) -> dict:
        return build_event_for_clip(grouped[clip_id], smooth_window=args.smooth_window)

    weak_rows = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for row in tqdm(
            executor.map(build_one, clip_ids),
            total=len(clip_ids),
            desc="Building Stage B weak events",
        ):
            weak_rows.append(row)

    write_weak_events(weak_rows)
    print(f"Wrote {len(weak_rows)} weak event row(s) to: {WEAK_EVENTS_CSV}")
    summarize(weak_rows)


if __name__ == "__main__":
    main()
