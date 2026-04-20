#!/usr/bin/env python3
"""
MLB Pitch Clip Processor (Motion-Based)

Detects the pitch moment using motion analysis and extracts focused frames.
Crops to the action zone and prepares data for CNN training.

Usage:
    python process_clips.py                  # Process all clips
    python process_clips.py --preview        # Preview without saving
    python process_clips.py --limit 10       # Process only N clips (for testing)
    python process_clips.py --debug          # Save debug visualizations
"""

import argparse
import json
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
NUM_FRAMES = 16          # Frames to extract from pitch window
IMAGE_SIZE = 128         # Output image size (smaller = faster training)
CROP_RATIO = 0.6         # Keep center 60% of frame (action zone)
MOTION_WINDOW = 30       # Frames to analyze for motion peak (~1 sec at 30fps)
MIN_GREEN_RATIO = 0.05   # Minimum green (grass) content to be valid pitch footage

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
        if pitch_type_dir.is_dir() and pitch_type_dir.name not in ("metadata.csv",):
            for clip_file in pitch_type_dir.glob("*.mp4"):
                clips.append(clip_file)
    return clips


def compute_motion_signal(frames: np.ndarray) -> np.ndarray:
    """Compute motion signal from frame differences.

    Args:
        frames: Array of shape (N, H, W, C) with uint8 values

    Returns:
        1D array of motion intensity per frame (length N-1)
    """
    # Convert to grayscale for faster computation
    gray_frames = np.array([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames])

    # Compute absolute differences between consecutive frames
    diffs = np.abs(gray_frames[1:].astype(np.float32) - gray_frames[:-1].astype(np.float32))

    # Sum motion per frame
    motion = diffs.sum(axis=(1, 2))

    # Smooth the signal with a small window
    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size
    motion_smooth = np.convolve(motion, kernel, mode='same')

    return motion_smooth


def find_pitch_window(motion_signal: np.ndarray, window_size: int = MOTION_WINDOW) -> tuple[int, int]:
    """Find the FIRST significant motion spike (the pitch, not the swing).

    The pitch always happens before the batter swings. We find the first
    window where motion exceeds a threshold, rather than the maximum
    (which might be the swing/hit).

    Args:
        motion_signal: 1D array of motion intensities
        window_size: Size of window to find

    Returns:
        (start_frame, end_frame) tuple
    """
    if len(motion_signal) < window_size:
        return 0, len(motion_signal)

    # Compute rolling sum for windows
    cumsum = np.cumsum(motion_signal)
    rolling_sum = cumsum[window_size:] - cumsum[:-window_size]

    # Find threshold: windows with motion above mean + 0.5*std are "significant"
    threshold = np.mean(rolling_sum) + 0.5 * np.std(rolling_sum)

    # Find FIRST window that exceeds threshold
    above_threshold = np.where(rolling_sum > threshold)[0]

    if len(above_threshold) > 0:
        # Take the first significant window
        peak_start = above_threshold[0]
    else:
        # Fallback to maximum if nothing exceeds threshold
        peak_start = np.argmax(rolling_sum)

    return peak_start, peak_start + window_size


def crop_to_action_zone(frame: np.ndarray, crop_ratio: float = CROP_RATIO) -> np.ndarray:
    """Crop frame to center action zone.

    Args:
        frame: Input frame (H, W, C)
        crop_ratio: Fraction of frame to keep (centered)

    Returns:
        Cropped frame
    """
    h, w = frame.shape[:2]

    # Calculate crop boundaries
    margin_x = int(w * (1 - crop_ratio) / 2)
    margin_y = int(h * (1 - crop_ratio) / 2)

    # Crop
    cropped = frame[margin_y:h-margin_y, margin_x:w-margin_x]

    return cropped


def check_green_content(frame_bgr: np.ndarray) -> float:
    """Check ratio of green (grass) pixels in a frame.

    Args:
        frame_bgr: Frame in BGR format (as read by OpenCV)

    Returns:
        Ratio of green pixels (0.0 to 1.0)
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    # Green hue range (grass)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    return np.sum(mask > 0) / mask.size


def read_all_frames(video_path: Path) -> np.ndarray | None:
    """Read all frames from a video file.

    Returns:
        Array of shape (N, H, W, 3) with uint8 RGB values, or None on failure
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return None

    frames = []
    frames_bgr = []  # Keep BGR versions for green check
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_bgr.append(frame)
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()

    if len(frames) < NUM_FRAMES:
        return None

    # Check green content on middle frame
    mid_frame = frames_bgr[len(frames_bgr) // 2]
    green_ratio = check_green_content(mid_frame)

    if green_ratio < MIN_GREEN_RATIO:
        return None  # Not valid pitch footage (close-up/replay)

    return np.array(frames)


def extract_pitch_frames(
    all_frames: np.ndarray,
    start_frame: int,
    num_frames: int = NUM_FRAMES,
    image_size: int = IMAGE_SIZE
) -> np.ndarray:
    """Extract and process frames from the pitch window.

    Args:
        all_frames: All video frames (N, H, W, C)
        start_frame: Starting frame index
        num_frames: Number of frames to extract
        image_size: Output image size

    Returns:
        Processed frames (num_frames, image_size, image_size, 3) normalized to 0-1
    """
    # Ensure we don't go past the end
    end_frame = min(start_frame + num_frames, len(all_frames))
    start_frame = max(0, end_frame - num_frames)

    # Extract frames
    pitch_frames = all_frames[start_frame:end_frame]

    # If we don't have enough frames, pad by repeating
    while len(pitch_frames) < num_frames:
        pitch_frames = np.concatenate([pitch_frames, pitch_frames[-1:]], axis=0)

    processed = []
    for frame in pitch_frames[:num_frames]:
        # Crop to action zone
        cropped = crop_to_action_zone(frame)
        # Resize
        resized = cv2.resize(cropped, (image_size, image_size))
        processed.append(resized)

    # Stack and normalize
    result = np.stack(processed, axis=0).astype(np.float32) / 255.0

    return result


def get_binary_label(pitch_type: str) -> str:
    """Convert pitch type to binary label."""
    if pitch_type.lower() in OFFSPEED_TYPES:
        return "offspeed"
    return "fastball"


def get_split(clip_id: str, metadata: dict) -> str:
    """Determine train/val/test split for a clip."""
    if clip_id not in metadata:
        return "train" if hash(clip_id) % 5 != 0 else "val"

    subset = metadata[clip_id].get("subset", "training")

    if subset == "testing":
        return "test"

    # 20% of training to validation
    if hash(clip_id) % 5 == 0:
        return "val"
    return "train"


def process_clip(
    clip_path: Path,
    metadata: dict,
    preview: bool = False,
    debug: bool = False
) -> dict | None:
    """Process a single clip with motion-based pitch detection.

    Returns:
        Info dict or None on failure
    """
    clip_id = clip_path.stem
    pitch_type = clip_path.parent.name
    binary_label = get_binary_label(pitch_type)
    split = get_split(clip_id, metadata)

    # Read all frames
    all_frames = read_all_frames(clip_path)
    if all_frames is None:
        return None

    # Compute motion signal
    motion_signal = compute_motion_signal(all_frames)

    # Find pitch window
    start_frame, end_frame = find_pitch_window(motion_signal)

    # Extract pitch frames
    frames = extract_pitch_frames(all_frames, start_frame)

    # Debug: save motion plot and sample frames
    if debug:
        save_debug_visualization(clip_id, all_frames, motion_signal, start_frame, end_frame, frames)

    if preview:
        return {
            "clip_id": clip_id,
            "original_type": pitch_type,
            "binary_label": binary_label,
            "split": split,
            "total_frames": len(all_frames),
            "pitch_window": (start_frame, end_frame),
            "output_shape": frames.shape,
            "saved": False
        }

    # Save
    out_path = OUTPUT_DIR / split / binary_label / f"{clip_id}.npy"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, frames)

    return {
        "clip_id": clip_id,
        "original_type": pitch_type,
        "binary_label": binary_label,
        "split": split,
        "total_frames": len(all_frames),
        "pitch_window": (start_frame, end_frame),
        "output_shape": frames.shape,
        "saved": True,
        "path": str(out_path)
    }


def save_debug_visualization(
    clip_id: str,
    all_frames: np.ndarray,
    motion_signal: np.ndarray,
    start_frame: int,
    end_frame: int,
    processed_frames: np.ndarray
):
    """Save debug visualization showing motion detection results."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    debug_dir = Path("debug")
    debug_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Top row: motion signal
    ax = axes[0, 0]
    ax.plot(motion_signal)
    ax.axvline(start_frame, color='r', linestyle='--', label='Pitch start')
    ax.axvline(end_frame, color='r', linestyle='--', label='Pitch end')
    ax.axvspan(start_frame, end_frame, alpha=0.3, color='red')
    ax.set_title(f'Motion Signal - {clip_id}')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Motion Intensity')
    ax.legend()

    # Show frames before, during, after pitch
    frame_indices = [
        max(0, start_frame - 20),  # Before pitch
        start_frame,                # Pitch start
        (start_frame + end_frame) // 2,  # Mid pitch
        min(len(all_frames)-1, end_frame + 10)  # After pitch
    ]

    for i, idx in enumerate(frame_indices):
        ax = axes[0, i] if i == 0 else axes[0, i]
        if i > 0:
            ax = axes[0, i]
        ax.imshow(all_frames[idx])
        labels = ['Before', 'Pitch Start', 'Mid Pitch', 'After']
        ax.set_title(f'{labels[i]} (frame {idx})')
        ax.axis('off')

    # Bottom row: processed frames
    frame_show_indices = [0, 5, 10, 15]
    for i, idx in enumerate(frame_show_indices):
        ax = axes[1, i]
        # Denormalize for display
        frame = (processed_frames[idx] * 255).astype(np.uint8)
        ax.imshow(frame)
        ax.set_title(f'Processed frame {idx}')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(debug_dir / f'{clip_id}_debug.png', dpi=100)
    plt.close()


def save_processing_metadata(results: list[dict], output_dir: Path):
    """Save processing metadata to CSV."""
    csv_path = output_dir / "metadata.csv"

    with open(csv_path, "w") as f:
        f.write("clip_id,original_type,binary_label,split,pitch_start,pitch_end\n")
        for r in results:
            start, end = r.get('pitch_window', (0, 0))
            f.write(f"{r['clip_id']},{r['original_type']},{r['binary_label']},{r['split']},{start},{end}\n")


def main():
    parser = argparse.ArgumentParser(description="Process MLB pitch clips with motion detection")
    parser.add_argument("--preview", action="store_true",
                        help="Preview processing without saving files")
    parser.add_argument("--limit", type=int,
                        help="Limit number of clips to process (for testing)")
    parser.add_argument("--debug", action="store_true",
                        help="Save debug visualizations")
    args = parser.parse_args()

    # Install matplotlib if debug mode and not installed
    if args.debug:
        try:
            import matplotlib
        except ImportError:
            print("Installing matplotlib for debug mode...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])

    # Load metadata
    print("Loading metadata...")
    metadata = load_metadata(METADATA_JSON)

    # Find clips
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
        result = process_clip(clip_path, metadata, args.preview, args.debug)

        if result:
            results.append(result)
        else:
            failed += 1

    # Summary
    print(f"\nProcessing complete!")
    print(f"  Successful: {len(results)}")
    print(f"  Failed: {failed}")

    if results:
        from collections import Counter
        split_counts = Counter(r["split"] for r in results)
        label_counts = Counter(r["binary_label"] for r in results)

        print(f"\nBy split:")
        for split, count in sorted(split_counts.items()):
            print(f"  {split}: {count}")

        print(f"\nBy label:")
        for label, count in sorted(label_counts.items()):
            print(f"  {label}: {count}")

        print(f"\nOutput shape: {results[0]['output_shape']}")

        # Show pitch detection stats
        windows = [r['pitch_window'] for r in results]
        starts = [w[0] for w in windows]
        print(f"\nPitch detection stats:")
        print(f"  Avg pitch start frame: {np.mean(starts):.1f}")
        print(f"  Range: {min(starts)} - {max(starts)}")

        if not args.preview:
            save_processing_metadata(results, OUTPUT_DIR)
            print(f"\nOutput saved to: {OUTPUT_DIR}/")

        if args.debug:
            print(f"Debug visualizations saved to: debug/")


if __name__ == "__main__":
    main()
