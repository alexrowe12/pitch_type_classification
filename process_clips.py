#!/usr/bin/env python3
"""
MLB Pitch Clip Processor

Finds the center-field broadcast segment, detects release and catch events,
and extracts a compact pitch sequence for training/debugging.

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
NUM_FRAMES = 6           # Frames to extract from release through catch
IMAGE_SIZE = 160         # Keep enough detail while staying lightweight
CROP_RATIO = 0.6         # Keep center 60% of frame (action zone)
MIN_GREEN_RATIO = 0.08   # Reject non-field shots more aggressively
VIEW_SCORE_THRESHOLD = 0.12
SHOT_CHANGE_THRESHOLD = 0.30
MIN_SEGMENT_FRAMES = 12
SEARCH_START_RATIO = 0.18
SEARCH_END_RATIO = 0.90
RELEASE_THRESHOLD_STD = 1.0
CATCH_THRESHOLD_STD = 0.9
MIN_RELEASE_TO_CATCH_GAP = 5
MAX_RELEASE_TO_CATCH_GAP = 18
MIN_PLATE_PEAK = 0.35
MAX_ROI_CORRELATION = 0.88
MIN_REGION_PEAK_SEPARATION = 0.75

# Pitch type mapping
OFFSPEED_TYPES = {"slider", "curveball", "changeup", "sinker", "knucklecurve"}


def smooth_signal(signal: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Smooth a 1D signal with edge padding to avoid suppressing endpoints."""
    if len(signal) == 0:
        return signal
    kernel_size = max(1, min(kernel_size, len(signal)))
    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
    pad_left = kernel_size // 2
    pad_right = kernel_size - 1 - pad_left
    padded = np.pad(signal.astype(np.float32), (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


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


def crop_region(frame: np.ndarray, top: float, bottom: float, left: float, right: float) -> np.ndarray:
    """Crop a region using relative coordinates."""
    h, w = frame.shape[:2]
    y1 = max(0, min(h - 1, int(round(h * top))))
    y2 = max(y1 + 1, min(h, int(round(h * bottom))))
    x1 = max(0, min(w - 1, int(round(w * left))))
    x2 = max(x1 + 1, min(w, int(round(w * right))))
    return frame[y1:y2, x1:x2]


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


def compute_motion_signal(frames: np.ndarray, region: tuple[float, float, float, float] | None = None) -> np.ndarray:
    """Compute a smoothed motion signal from frame differences."""
    roi_frames = frames if region is None else np.array([crop_region(frame, *region) for frame in frames])
    gray_frames = np.array([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in roi_frames])
    diffs = np.abs(gray_frames[1:].astype(np.float32) - gray_frames[:-1].astype(np.float32))
    motion = diffs.mean(axis=(1, 2))
    return smooth_signal(motion, kernel_size=5)


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


def compute_view_scores(frames_bgr: list[np.ndarray]) -> np.ndarray:
    """Score each frame for likelihood of being the center-field pitch camera."""
    scores = []
    for frame in frames_bgr:
        lower_green = check_green_content(crop_region(frame, 0.42, 1.0, 0.0, 1.0))
        center_green = check_green_content(crop_region(frame, 0.32, 0.92, 0.18, 0.82))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vertical_edges = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        edge_strength = float(np.mean(np.abs(vertical_edges)))

        score = 0.55 * lower_green + 0.45 * center_green - 0.0008 * edge_strength
        scores.append(score)

    return smooth_signal(np.array(scores, dtype=np.float32), kernel_size=7)


def compute_shot_change_signal(frames_bgr: list[np.ndarray]) -> np.ndarray:
    """Estimate hard cuts using HSV histogram deltas between consecutive frames."""
    if len(frames_bgr) < 2:
        return np.array([], dtype=np.float32)

    histograms = []
    for frame in frames_bgr:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        histograms.append(hist)

    deltas = []
    for previous, current in zip(histograms[:-1], histograms[1:]):
        deltas.append(float(cv2.compareHist(previous, current, cv2.HISTCMP_BHATTACHARYYA)))

    return np.array(deltas, dtype=np.float32)


def find_candidate_segments(view_scores: np.ndarray, shot_changes: np.ndarray) -> list[tuple[int, int]]:
    """Find contiguous frame segments that likely contain the pitch broadcast angle."""
    if len(view_scores) == 0:
        return []

    frame_mask = view_scores >= VIEW_SCORE_THRESHOLD
    if len(shot_changes) > 0:
        cut_mask = np.concatenate([[False], shot_changes > SHOT_CHANGE_THRESHOLD])
        frame_mask = frame_mask & ~cut_mask

    segments = []
    start = None
    for idx, keep in enumerate(frame_mask):
        if keep and start is None:
            start = idx
        elif not keep and start is not None:
            if idx - start >= MIN_SEGMENT_FRAMES:
                segments.append((start, idx - 1))
            start = None

    if start is not None and len(frame_mask) - start >= MIN_SEGMENT_FRAMES:
        segments.append((start, len(frame_mask) - 1))

    return segments


def find_local_peaks(signal: np.ndarray, start: int, end: int, threshold_std: float) -> list[int]:
    """Return local peaks above a z-score-like threshold within a range."""
    if len(signal) == 0:
        return []

    start = max(1, start)
    end = min(len(signal) - 1, end)
    if end <= start:
        return []

    window = signal[start:end]
    threshold = float(np.mean(window) + threshold_std * np.std(window))
    peaks = []
    for idx in range(start, end):
        if signal[idx] >= threshold and signal[idx] >= signal[idx - 1] and signal[idx] >= signal[idx + 1]:
            peaks.append(idx)
    return peaks


def choose_best_segment(segments: list[tuple[int, int]], view_scores: np.ndarray) -> tuple[int, int] | None:
    """Pick the most promising segment using quality and duration."""
    if not segments:
        return None

    def segment_score(segment: tuple[int, int]) -> float:
        start, end = segment
        quality = float(np.mean(view_scores[start:end + 1]))
        length = end - start + 1
        return quality * 100.0 + length

    return max(segments, key=segment_score)


def score_segment_event(
    segment_start: int,
    segment_end: int,
    segment_frames: np.ndarray,
    view_scores: np.ndarray,
) -> dict | None:
    """Score one segment for a plausible pitcher-release to catcher-catch sequence."""
    pitcher_motion = compute_motion_signal(segment_frames, region=(0.44, 0.90, 0.12, 0.42))
    plate_motion = compute_motion_signal(segment_frames, region=(0.34, 0.84, 0.46, 0.82))
    lane_motion = compute_motion_signal(segment_frames, region=(0.28, 0.82, 0.22, 0.82))

    if len(pitcher_motion) == 0 or len(plate_motion) == 0:
        return None

    search_start = max(1, int(len(pitcher_motion) * SEARCH_START_RATIO))
    search_end = min(len(pitcher_motion) - 1, int(len(pitcher_motion) * SEARCH_END_RATIO))
    release_candidates = find_local_peaks(pitcher_motion, search_start, search_end, RELEASE_THRESHOLD_STD)

    if release_candidates:
        release_local = release_candidates[0]
    else:
        release_local = search_start + int(np.argmax(pitcher_motion[search_start:search_end]))

    catch_start = min(len(plate_motion) - 1, release_local + MIN_RELEASE_TO_CATCH_GAP)
    catch_end = min(len(plate_motion) - 1, release_local + MAX_RELEASE_TO_CATCH_GAP)
    catch_candidates = find_local_peaks(plate_motion, catch_start, catch_end, CATCH_THRESHOLD_STD)

    if catch_candidates:
        catch_local = catch_candidates[0]
    else:
        if catch_end <= catch_start:
            catch_local = min(len(plate_motion) - 1, release_local + MIN_RELEASE_TO_CATCH_GAP)
        else:
            catch_local = catch_start + int(np.argmax(plate_motion[catch_start:catch_end]))

    if catch_local <= release_local:
        catch_local = min(len(segment_frames) - 1, release_local + MIN_RELEASE_TO_CATCH_GAP)

    release_frame = segment_start + release_local
    catch_frame = segment_start + catch_local + 1
    sample_frames = np.linspace(release_frame, catch_frame, num=NUM_FRAMES)
    sample_frames = np.clip(np.round(sample_frames).astype(int), 0, segment_end)

    pitcher_peak = float(pitcher_motion[release_local])
    plate_peak = float(plate_motion[min(catch_local, len(plate_motion) - 1)])
    lane_peak = float(np.max(lane_motion[release_local:min(len(lane_motion), catch_local + 1)]))
    pitcher_baseline = float(np.mean(pitcher_motion[:max(1, release_local)]))
    plate_baseline = float(np.mean(plate_motion[:max(1, release_local)]))
    roi_length = min(len(pitcher_motion), len(plate_motion))
    roi_correlation = float(np.corrcoef(pitcher_motion[:roi_length], plate_motion[:roi_length])[0, 1]) if roi_length >= 2 else 0.0
    region_peak_separation = abs(pitcher_peak - plate_peak)

    if plate_peak < MIN_PLATE_PEAK:
        return None
    if roi_correlation > MAX_ROI_CORRELATION and region_peak_separation < MIN_REGION_PEAK_SEPARATION:
        return None

    event_score = (
        2.2 * (pitcher_peak - pitcher_baseline)
        + 2.0 * (plate_peak - plate_baseline)
        + 1.0 * lane_peak
        + 40.0 * float(np.mean(view_scores[segment_start:segment_end + 1]))
        - 0.15 * max(0, (catch_frame - release_frame) - 12)
    )

    return {
        "release_frame": int(release_frame),
        "catch_frame": int(catch_frame),
        "sample_frames": sample_frames.tolist(),
        "segment": (int(segment_start), int(segment_end)),
        "pitcher_motion": pitcher_motion,
        "plate_motion": plate_motion,
        "lane_motion": lane_motion,
        "release_local": int(release_local),
        "catch_local": int(catch_local),
        "event_score": float(event_score),
        "pitcher_peak": pitcher_peak,
        "plate_peak": plate_peak,
        "lane_peak": lane_peak,
        "roi_correlation": roi_correlation,
        "region_peak_separation": region_peak_separation,
    }


def detect_pitch_window(all_frames: np.ndarray, all_frames_bgr: list[np.ndarray]) -> dict | None:
    """Detect release and catch frames inside the best-scoring segment."""
    if len(all_frames) < NUM_FRAMES:
        return None

    view_scores = compute_view_scores(all_frames_bgr)
    shot_changes = compute_shot_change_signal(all_frames_bgr)
    segments = find_candidate_segments(view_scores, shot_changes)
    if not segments:
        return None

    best_detection = None
    for segment_start, segment_end in segments:
        segment_frames = all_frames[segment_start:segment_end + 1]
        detection = score_segment_event(segment_start, segment_end, segment_frames, view_scores)
        if detection is None:
            continue
        if best_detection is None or detection["event_score"] > best_detection["event_score"]:
            best_detection = detection

    if best_detection is None:
        fallback_segment = choose_best_segment(segments, view_scores)
        if fallback_segment is None:
            return None
        segment_start, segment_end = fallback_segment
        segment_frames = all_frames[segment_start:segment_end + 1]
        best_detection = score_segment_event(segment_start, segment_end, segment_frames, view_scores)
        if best_detection is None:
            return None

    best_detection["view_scores"] = view_scores
    best_detection["shot_changes"] = shot_changes
    best_detection["segments"] = segments
    return best_detection


def read_all_frames(video_path: Path) -> tuple[np.ndarray, list[np.ndarray]] | None:
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

    if max(check_green_content(frame) for frame in frames_bgr) < MIN_GREEN_RATIO:
        return None

    return np.array(frames), frames_bgr


def enhance_ball_visibility(frame: np.ndarray) -> np.ndarray:
    """Preserve small bright details in the action zone."""
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)
    enhanced_bgr = cv2.cvtColor(cv2.merge((l_enhanced, a_channel, b_channel)), cv2.COLOR_LAB2BGR)

    hsv = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2HSV)
    bright_mask = cv2.inRange(hsv, np.array([0, 0, 170]), np.array([180, 70, 255]))
    bright_mask = cv2.GaussianBlur(bright_mask, (5, 5), 0).astype(np.float32) / 255.0

    sharpened = cv2.addWeighted(
        enhanced_bgr,
        1.25,
        cv2.GaussianBlur(enhanced_bgr, (0, 0), 1.2),
        -0.25,
        0
    )
    blended = enhanced_bgr.astype(np.float32) * (1.0 - 0.35 * bright_mask[..., None])
    blended += sharpened.astype(np.float32) * (0.35 * bright_mask[..., None])

    return cv2.cvtColor(np.clip(blended, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)


def extract_pitch_frames(
    all_frames: np.ndarray,
    sample_frames: list[int],
    num_frames: int = NUM_FRAMES,
    image_size: int = IMAGE_SIZE
) -> np.ndarray:
    """Extract and process frames from the pitch window.

    Args:
        all_frames: All video frames (N, H, W, C)
        sample_frames: Specific frame indices to extract
        num_frames: Number of frames to extract
        image_size: Output image size

    Returns:
        Processed frames (num_frames, image_size, image_size, 3) normalized to 0-1
    """
    pitch_frames = []
    for idx in sample_frames[:num_frames]:
        idx = max(0, min(len(all_frames) - 1, int(idx)))
        pitch_frames.append(all_frames[idx])

    while len(pitch_frames) < num_frames:
        pitch_frames.append(pitch_frames[-1])

    processed = []
    for frame in pitch_frames[:num_frames]:
        # Crop to action zone
        cropped = crop_to_action_zone(frame)
        enhanced = enhance_ball_visibility(cropped)
        # Resize
        resized = cv2.resize(enhanced, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
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

    frame_data = read_all_frames(clip_path)
    if frame_data is None:
        return None
    all_frames, all_frames_bgr = frame_data

    detection = detect_pitch_window(all_frames, all_frames_bgr)
    if detection is None:
        return None

    frames = extract_pitch_frames(all_frames, detection["sample_frames"])

    # Debug: save motion plot and sample frames
    if debug:
        save_debug_visualization(clip_id, all_frames, detection, frames)

    if preview:
        return {
            "clip_id": clip_id,
            "original_type": pitch_type,
            "binary_label": binary_label,
            "split": split,
            "total_frames": len(all_frames),
            "pitch_window": (detection["release_frame"], detection["catch_frame"]),
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
        "pitch_window": (detection["release_frame"], detection["catch_frame"]),
        "output_shape": frames.shape,
        "saved": True,
        "path": str(out_path)
    }


def save_debug_visualization(
    clip_id: str,
    all_frames: np.ndarray,
    detection: dict,
    processed_frames: np.ndarray
):
    """Save debug visualization showing motion detection results."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    debug_dir = Path("debug")
    debug_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(4, 4, figsize=(18, 14))

    release_frame = detection["release_frame"]
    catch_frame = detection["catch_frame"]
    segment_start, segment_end = detection["segment"]
    sample_frames = detection["sample_frames"]

    ax = axes[0, 0]
    ax.plot(detection["view_scores"], label="View score")
    for start, end in detection["segments"]:
        ax.axvspan(start, end, color="green", alpha=0.12)
    ax.axvline(segment_start, color="green", linestyle="--", label="Best segment")
    ax.axvline(segment_end, color="green", linestyle="--")
    ax.axvline(release_frame, color="red", linestyle="--", label="Release")
    ax.axvline(catch_frame, color="blue", linestyle="--", label="Catch")
    ax.set_title(f"View score - {clip_id}")
    ax.set_xlabel("Frame")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[0, 1]
    if len(detection["shot_changes"]) > 0:
        ax.plot(np.arange(1, len(detection["shot_changes"]) + 1), detection["shot_changes"], label="Shot change")
        ax.axhline(SHOT_CHANGE_THRESHOLD, color="orange", linestyle="--", label="Cut threshold")
    ax.set_title("Shot change signal")
    ax.set_xlabel("Frame")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[0, 2]
    local_x = np.arange(segment_start + 1, segment_start + 1 + len(detection["pitcher_motion"]))
    ax.plot(local_x, detection["pitcher_motion"], label="Pitcher motion")
    ax.plot(local_x, detection["plate_motion"], label="Plate motion")
    ax.plot(local_x, detection["lane_motion"], label="Lane motion")
    ax.axvline(release_frame, color="red", linestyle="--")
    ax.axvline(catch_frame, color="blue", linestyle="--")
    ax.set_title("ROI motion signals")
    ax.set_xlabel("Frame")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[0, 3]
    ax.axis("off")
    ax.text(
        0.02,
        0.98,
        "\n".join(
            [
                f"event_score={detection['event_score']:.2f}",
                f"pitcher_peak={detection['pitcher_peak']:.2f}",
                f"plate_peak={detection['plate_peak']:.2f}",
                f"lane_peak={detection['lane_peak']:.2f}",
                f"roi_corr={detection['roi_correlation']:.2f}",
                f"peak_sep={detection['region_peak_separation']:.2f}",
                f"segment={segment_start}-{segment_end}",
                f"release={release_frame}",
                f"catch={catch_frame}",
            ]
        ),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
    )

    context_frames = [
        max(0, release_frame - 3),
        release_frame,
        min(len(all_frames) - 1, (release_frame + catch_frame) // 2),
        catch_frame,
    ]
    context_labels = ["Before", "Release", "Mid flight", "Catch"]
    for i, (label, idx) in enumerate(zip(context_labels, context_frames)):
        ax = axes[1, i]
        ax.imshow(all_frames[idx])
        ax.set_title(f"{label} ({idx})")
        ax.axis("off")

    for i, idx in enumerate(sample_frames):
        row = 2 + (i // 4)
        col = i % 4
        ax = axes[row, col]
        frame = (processed_frames[i] * 255).astype(np.uint8)
        ax.imshow(frame)
        ax.set_title(f"Sample {i} -> src {idx}")
        ax.axis("off")

    if NUM_FRAMES <= 4:
        for col in range(NUM_FRAMES, 4):
            axes[2, col].axis("off")
    else:
        for col in range(NUM_FRAMES - 4, 4):
            axes[3, col].axis("off")

    fig.suptitle(
        f"segment={segment_start}-{segment_end} release={release_frame} catch={catch_frame}",
        fontsize=12,
    )

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
