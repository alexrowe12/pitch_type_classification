#!/usr/bin/env python3
"""
Generate weak labels for Stage A frame classification.

Usage:
    python -m stage_a.build_weak_labels
    python -m stage_a.build_weak_labels --limit 500
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from stage_a.paths import (
    LABELS_DIR,
    WEAK_LABELS_CSV,
    ensure_stage_a_dirs,
)


FRAME_EXPORTS_CSV = LABELS_DIR / "frame_exports.csv"


def load_frame_exports(limit: int | None = None) -> list[dict]:
    """Load exported frame metadata rows."""
    if not FRAME_EXPORTS_CSV.exists():
        raise FileNotFoundError(f"Missing frame export metadata: {FRAME_EXPORTS_CSV}")

    with open(FRAME_EXPORTS_CSV, newline="") as handle:
        rows = list(csv.DictReader(handle))

    if limit is not None:
        rows = rows[:limit]
    return rows


def green_ratio(frame_bgr: np.ndarray) -> float:
    """Estimate how much of the frame is grass-like green."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([35, 40, 35], dtype=np.uint8)
    upper = np.array([90, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    return float(np.mean(mask > 0))


def compute_features(frame_bgr: np.ndarray) -> dict[str, float]:
    """Compute simple view-layout features used for weak supervision."""
    h, w = frame_bgr.shape[:2]

    lower_half = frame_bgr[int(h * 0.45):, :]
    center_band = frame_bgr[int(h * 0.25):int(h * 0.85), int(w * 0.18):int(w * 0.82)]
    upper_center = frame_bgr[:int(h * 0.35), int(w * 0.25):int(w * 0.75)]

    frame_green = green_ratio(frame_bgr)
    lower_green = green_ratio(lower_half)
    center_green = green_ratio(center_band)
    upper_center_green = green_ratio(upper_center)

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(np.mean(edges > 0))

    brightness = float(np.mean(gray) / 255.0)
    saturation = float(np.mean(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)[:, :, 1]) / 255.0)

    return {
        "frame_green": frame_green,
        "lower_green": lower_green,
        "center_green": center_green,
        "upper_center_green": upper_center_green,
        "edge_density": edge_density,
        "brightness": brightness,
        "saturation": saturation,
    }


def assign_weak_label(features: dict[str, float]) -> tuple[str, float, str]:
    """Assign a conservative weak label and confidence from image heuristics."""
    lower_green = features["lower_green"]
    center_green = features["center_green"]
    frame_green = features["frame_green"]
    upper_center_green = features["upper_center_green"]
    edge_density = features["edge_density"]

    if lower_green >= 0.18 and center_green >= 0.08 and edge_density >= 0.035:
        confidence = min(0.95, 0.55 + lower_green + 0.5 * center_green)
        return "pitch_camera", confidence, "field_layout_rule"

    if frame_green <= 0.22 and lower_green >= 0.20 and center_green >= 0.12 and upper_center_green >= 0.08 and edge_density <= 0.034:
        confidence = 0.72
        return "non_pitch_camera", confidence, "closeup_with_padding_rule"

    if frame_green <= 0.05 and lower_green <= 0.08 and edge_density <= 0.06:
        confidence = min(0.95, 0.60 + (0.08 - lower_green) + (0.05 - frame_green))
        return "non_pitch_camera", confidence, "closeup_rule"

    if lower_green <= 0.10 and center_green <= 0.05:
        confidence = 0.58
        return "non_pitch_camera", confidence, "low_field_rule"

    return "unknown", 0.0, "no_rule"


def build_rows(export_rows: list[dict]) -> list[dict]:
    """Create weak-label rows from exported-frame metadata."""
    weak_rows = []
    for row in export_rows:
        frame_path = Path(row["frame_path"])
        frame_bgr = cv2.imread(str(frame_path))
        if frame_bgr is None:
            continue

        features = compute_features(frame_bgr)
        weak_label, weak_confidence, weak_source = assign_weak_label(features)

        weak_rows.append(
            {
                "clip_id": row["clip_id"],
                "frame_idx": row["frame_idx"],
                "frame_path": row["frame_path"],
                "pitch_type": row["pitch_type"],
                "weak_label": weak_label,
                "weak_confidence": f"{weak_confidence:.4f}",
                "weak_source": weak_source,
                "frame_green": f"{features['frame_green']:.4f}",
                "lower_green": f"{features['lower_green']:.4f}",
                "center_green": f"{features['center_green']:.4f}",
                "upper_center_green": f"{features['upper_center_green']:.4f}",
                "edge_density": f"{features['edge_density']:.4f}",
                "brightness": f"{features['brightness']:.4f}",
                "saturation": f"{features['saturation']:.4f}",
            }
        )

    return weak_rows


def write_weak_labels(rows: list[dict]) -> None:
    """Write weak labels to CSV."""
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "clip_id",
        "frame_idx",
        "frame_path",
        "pitch_type",
        "weak_label",
        "weak_confidence",
        "weak_source",
        "frame_green",
        "lower_green",
        "center_green",
        "upper_center_green",
        "edge_density",
        "brightness",
        "saturation",
    ]

    with open(WEAK_LABELS_CSV, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> None:
    """Print a compact label summary."""
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["weak_label"]] = counts.get(row["weak_label"], 0) + 1

    print("Weak-label summary:")
    for label in sorted(counts):
        print(f"  {label}: {counts[label]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build weak Stage A labels from exported frames")
    parser.add_argument("--limit", type=int, help="Limit the number of exported frames processed")
    args = parser.parse_args()

    ensure_stage_a_dirs()

    export_rows = load_frame_exports(limit=args.limit)
    print(f"Loaded {len(export_rows)} exported frame row(s)")

    weak_rows = build_rows(export_rows)
    write_weak_labels(weak_rows)

    print(f"Wrote weak labels to: {WEAK_LABELS_CSV}")
    summarize(weak_rows)


if __name__ == "__main__":
    main()
