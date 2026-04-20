#!/usr/bin/env python3
"""
MLB Pitch Clip Processor

Extracts frames from video clips and prepares them for CNN training.
Converts pitch types to binary classification: fastball vs off-speed.

Usage:
    python process_clips.py                  # Process all clips
    python process_clips.py --frames 16      # Custom frame count
    python process_clips.py --preview        # Preview without saving
    python process_clips.py --limit 10       # Process only N clips (for testing)
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# Paths
CLIPS_DIR = Path("clips")
OUTPUT_DIR = Path("processed")
METADATA_JSON = "mlb-youtube-repo/data/mlb-youtube-segmented.json"

# Processing settings
DEFAULT_FRAMES = 16
IMAGE_SIZE = 224

# Pitch type mapping
OFFSPEED_TYPES = {"slider", "curveball", "changeup", "sinker", "knucklecurve"}


def load_metadata(json_path: str) -> dict:
    """Load the original dataset metadata for train/test split info."""
    with open(json_path, "r") as f:
        return json.load(f)


def get_clip_files() -> list[Path]:
    """Find all downloaded clip files."""
    clips = []
    for pitch_type_dir in CLIPS_DIR.iterdir():
        if pitch_type_dir.is_dir() and pitch_type_dir.name != "metadata.csv":
            for clip_file in pitch_type_dir.glob("*.mp4"):
                clips.append(clip_file)
    return clips


def extract_frames(video_path: Path, num_frames: int) -> np.ndarray | None:
    """Extract evenly-spaced frames from a video clip.

    Returns:
        np.ndarray of shape (num_frames, IMAGE_SIZE, IMAGE_SIZE, 3) or None on failure
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < num_frames:
        cap.release()
        return None

    # Calculate frame indices to extract (evenly spaced)
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret:
            cap.release()
            return None

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize to target size
        frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))

        frames.append(frame)

    cap.release()

    # Stack frames and normalize to 0-1
    frames_array = np.stack(frames, axis=0).astype(np.float32) / 255.0

    return frames_array


def get_binary_label(pitch_type: str) -> str:
    """Convert pitch type to binary label."""
    if pitch_type.lower() in OFFSPEED_TYPES:
        return "offspeed"
    return "fastball"


def get_split(clip_id: str, metadata: dict, val_ratio: float = 0.2) -> str:
    """Determine train/val/test split for a clip.

    - Test clips stay in test
    - Training clips are split 80/20 into train/val (deterministically by clip_id)
    """
    if clip_id not in metadata:
        # If not in metadata, hash the clip_id to determine split
        return "train" if hash(clip_id) % 5 != 0 else "val"

    subset = metadata[clip_id].get("subset", "training")

    if subset == "testing":
        return "test"

    # Deterministically split training into train/val based on clip_id
    # Use hash for reproducibility
    if hash(clip_id) % 5 == 0:  # ~20% to validation
        return "val"
    return "train"


def process_clip(
    clip_path: Path,
    metadata: dict,
    num_frames: int,
    output_dir: Path,
    preview: bool = False
) -> dict | None:
    """Process a single clip. Returns info dict or None on failure."""
    clip_id = clip_path.stem
    pitch_type = clip_path.parent.name
    binary_label = get_binary_label(pitch_type)
    split = get_split(clip_id, metadata)

    # Extract frames
    frames = extract_frames(clip_path, num_frames)

    if frames is None:
        return None

    if preview:
        return {
            "clip_id": clip_id,
            "original_type": pitch_type,
            "binary_label": binary_label,
            "split": split,
            "shape": frames.shape,
            "saved": False
        }

    # Save to output directory
    out_path = output_dir / split / binary_label / f"{clip_id}.npy"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, frames)

    return {
        "clip_id": clip_id,
        "original_type": pitch_type,
        "binary_label": binary_label,
        "split": split,
        "shape": frames.shape,
        "saved": True,
        "path": str(out_path)
    }


def save_processing_metadata(results: list[dict], output_dir: Path):
    """Save processing metadata to CSV."""
    csv_path = output_dir / "metadata.csv"

    with open(csv_path, "w") as f:
        f.write("clip_id,original_type,binary_label,split\n")
        for r in results:
            f.write(f"{r['clip_id']},{r['original_type']},{r['binary_label']},{r['split']}\n")


def main():
    parser = argparse.ArgumentParser(description="Process MLB pitch video clips")
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES,
                        help=f"Number of frames to extract per clip (default: {DEFAULT_FRAMES})")
    parser.add_argument("--preview", action="store_true",
                        help="Preview processing without saving files")
    parser.add_argument("--limit", type=int,
                        help="Limit number of clips to process (for testing)")
    args = parser.parse_args()

    # Load metadata for split information
    print("Loading metadata...")
    metadata = load_metadata(METADATA_JSON)

    # Find all clips
    clips = get_clip_files()
    print(f"Found {len(clips)} clips to process")

    if args.limit:
        clips = clips[:args.limit]
        print(f"Limited to {len(clips)} clips")

    if args.preview:
        print("\n** PREVIEW MODE - no files will be saved **\n")

    # Process clips
    results = []
    failed = 0

    for clip_path in tqdm(clips, desc="Processing clips"):
        result = process_clip(clip_path, metadata, args.frames, OUTPUT_DIR, args.preview)

        if result:
            results.append(result)
        else:
            failed += 1

    # Summary
    print(f"\nProcessing complete!")
    print(f"  Successful: {len(results)}")
    print(f"  Failed: {failed}")

    if results:
        # Count by split and label
        from collections import Counter
        split_counts = Counter(r["split"] for r in results)
        label_counts = Counter(r["binary_label"] for r in results)

        print(f"\nBy split:")
        for split, count in sorted(split_counts.items()):
            print(f"  {split}: {count}")

        print(f"\nBy label:")
        for label, count in sorted(label_counts.items()):
            print(f"  {label}: {count}")

        print(f"\nFrame shape: {results[0]['shape']}")

        if not args.preview:
            save_processing_metadata(results, OUTPUT_DIR)
            print(f"\nOutput saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
