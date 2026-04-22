#!/usr/bin/env python3
"""
Run Stage A inference and produce clip-level pitch-camera segments.

Usage:
    python -m stage_a.infer_stage_a
    python -m stage_a.infer_stage_a --threshold 0.65
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms

from stage_a.paths import (
    CLIP_SEGMENTS_CSV,
    FRAME_PREDICTIONS_CSV,
    STAGE_A_MODEL_PT,
    LABELS_DIR,
    ensure_stage_a_dirs,
)


LABEL_TO_INDEX = {"non_pitch_camera": 0, "pitch_camera": 1}
IMAGE_SIZE = 224
FRAME_EXPORTS_CSV = LABELS_DIR / "frame_exports.csv"


def load_export_rows(path: Path) -> list[dict]:
    """Load exported frame rows for inference."""
    if not path.exists():
        raise FileNotFoundError(f"Missing frame exports at {path}. Run stage_a.export_frames first.")
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def build_model() -> torch.nn.Module:
    """Rebuild the trained model architecture."""
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, len(LABEL_TO_INDEX))
    return model


def build_transform():
    """Return the eval transform."""
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def predict_rows(rows: list[dict], model: torch.nn.Module, device: torch.device) -> list[dict]:
    """Run per-frame inference for all rows."""
    transform = build_transform()
    predictions = []

    model.eval()
    with torch.no_grad():
        for row in rows:
            image = Image.open(row["frame_path"]).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().squeeze(0)
            pitch_prob = float(probs[LABEL_TO_INDEX["pitch_camera"]])

            predictions.append(
                {
                    "clip_id": row["clip_id"],
                    "frame_idx": int(row["frame_idx"]),
                    "frame_path": row["frame_path"],
                    "pitch_type": row["pitch_type"],
                    "pitch_camera_probability": pitch_prob,
                }
            )

    return predictions


def smooth_probabilities(values: list[float], window_size: int = 3) -> list[float]:
    """Apply a simple moving average to probabilities."""
    if not values:
        return []
    if window_size <= 1 or len(values) < 2:
        return values

    smoothed = []
    for idx in range(len(values)):
        start = max(0, idx - window_size // 2)
        end = min(len(values), idx + window_size // 2 + 1)
        smoothed.append(sum(values[start:end]) / (end - start))
    return smoothed


def choose_best_segment(predictions: list[dict], threshold: float) -> dict | None:
    """Choose the best contiguous high-confidence segment for one clip."""
    if not predictions:
        return None

    frame_indices = [row["frame_idx"] for row in predictions]
    probs = [row["pitch_camera_probability"] for row in predictions]
    smoothed = smooth_probabilities(probs)

    segments = []
    current_start = None
    current_scores = []
    current_frames = []

    for frame_idx, score in zip(frame_indices, smoothed):
        if score >= threshold:
            if current_start is None:
                current_start = frame_idx
            current_frames.append(frame_idx)
            current_scores.append(score)
        else:
            if current_start is not None:
                segments.append((current_frames[0], current_frames[-1], current_scores))
                current_start = None
                current_frames = []
                current_scores = []

    if current_start is not None:
        segments.append((current_frames[0], current_frames[-1], current_scores))

    if not segments:
        best_idx = max(range(len(smoothed)), key=lambda idx: smoothed[idx])
        return {
            "segment_start_frame": frame_indices[best_idx],
            "segment_end_frame": frame_indices[best_idx],
            "segment_mean_probability": smoothed[best_idx],
            "segment_num_sampled_frames": 1,
        }

    def segment_score(segment):
        start, end, scores = segment
        return (sum(scores) / len(scores)) * len(scores)

    best = max(segments, key=segment_score)
    return {
        "segment_start_frame": best[0],
        "segment_end_frame": best[1],
        "segment_mean_probability": sum(best[2]) / len(best[2]),
        "segment_num_sampled_frames": len(best[2]),
    }


def write_frame_predictions(rows: list[dict]) -> None:
    """Write per-frame probabilities CSV."""
    FRAME_PREDICTIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "clip_id",
        "frame_idx",
        "frame_path",
        "pitch_type",
        "pitch_camera_probability",
    ]
    with open(FRAME_PREDICTIONS_CSV, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "pitch_camera_probability": f"{row['pitch_camera_probability']:.6f}",
                }
            )


def write_clip_segments(rows: list[dict]) -> None:
    """Write clip-level segment selection CSV."""
    CLIP_SEGMENTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "clip_id",
        "pitch_type",
        "segment_start_frame",
        "segment_end_frame",
        "segment_mean_probability",
        "segment_num_sampled_frames",
    ]
    with open(CLIP_SEGMENTS_CSV, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "segment_mean_probability": f"{row['segment_mean_probability']:.6f}",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage A inference over exported frames")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="Probability threshold for selecting pitch-camera frames into segments",
    )
    args = parser.parse_args()

    ensure_stage_a_dirs()
    if not STAGE_A_MODEL_PT.exists():
        raise FileNotFoundError(
            f"Missing trained model at {STAGE_A_MODEL_PT}. Run stage_a.train_stage_a first."
        )

    rows = load_export_rows(FRAME_EXPORTS_CSV)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model().to(device)
    state_dict = torch.load(STAGE_A_MODEL_PT, map_location=device)
    model.load_state_dict(state_dict)

    frame_predictions = predict_rows(rows, model, device)
    write_frame_predictions(frame_predictions)

    by_clip: dict[str, list[dict]] = defaultdict(list)
    for row in frame_predictions:
        by_clip[row["clip_id"]].append(row)

    clip_segments = []
    for clip_id, clip_rows in sorted(by_clip.items()):
        sorted_rows = sorted(clip_rows, key=lambda row: row["frame_idx"])
        segment = choose_best_segment(sorted_rows, threshold=args.threshold)
        if segment is None:
            continue
        clip_segments.append(
            {
                "clip_id": clip_id,
                "pitch_type": sorted_rows[0]["pitch_type"],
                **segment,
            }
        )

    write_clip_segments(clip_segments)
    print(f"Wrote frame predictions to: {FRAME_PREDICTIONS_CSV}")
    print(f"Wrote clip segments to: {CLIP_SEGMENTS_CSV}")


if __name__ == "__main__":
    main()
