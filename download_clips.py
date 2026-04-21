#!/usr/bin/env python3
"""
MLB Pitch Clip Downloader

Downloads pitch video clips from YouTube and organizes them by pitch type.
Uses yt-dlp to download specific segments directly (no full video downloads).

Usage:
    python download_clips.py --limit 100    # Test with 100 clips
    python download_clips.py                 # Download all clips
    python download_clips.py --types fastball slider  # Specific pitch types
"""

import argparse
import json
import subprocess
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    print("Installing tqdm...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
    from tqdm import tqdm


# Paths
DATA_JSON = "mlb-youtube-repo/data/mlb-youtube-segmented.json"
OUTPUT_DIR = Path("clips")
ERROR_LOG = "download_errors.log"
DEFAULT_WORKERS = 8

ERROR_LOG_LOCK = threading.Lock()


def check_dependencies():
    """Verify yt-dlp and ffmpeg are installed."""
    missing = []

    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        missing.append("yt-dlp (install with: pip install yt-dlp)")

    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        missing.append("ffmpeg (install with: brew install ffmpeg)")

    if missing:
        print("Missing dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        sys.exit(1)


def load_dataset(json_path: str) -> dict:
    """Load the MLB YouTube dataset."""
    with open(json_path, "r") as f:
        return json.load(f)


def stratified_sample(data: dict, limit: int) -> dict:
    """Sample clips maintaining pitch type distribution."""
    # Group by pitch type
    by_type = defaultdict(list)
    for clip_id, clip in data.items():
        pitch_type = clip.get("type", "unknown")
        by_type[pitch_type].append((clip_id, clip))

    # Calculate proportional samples per type
    total = len(data)
    sampled = {}
    remaining = limit

    # Sort types by count (descending) to handle rounding
    sorted_types = sorted(by_type.items(), key=lambda x: -len(x[1]))

    for i, (pitch_type, clips) in enumerate(sorted_types):
        if i == len(sorted_types) - 1:
            # Last type gets remaining slots
            n_samples = min(remaining, len(clips))
        else:
            # Proportional allocation
            proportion = len(clips) / total
            n_samples = min(int(limit * proportion), len(clips), remaining)

        # Take first n_samples (could randomize, but deterministic is nice for testing)
        for clip_id, clip in clips[:n_samples]:
            sampled[clip_id] = clip
        remaining -= n_samples

    return sampled


def filter_valid_clips(data: dict) -> tuple[dict, int]:
    """Filter out clips with bad data (unknown pitch type, negative timestamps).
    Returns (filtered_data, num_removed)."""
    valid = {}
    for clip_id, clip in data.items():
        pitch_type = clip.get("type", "unknown")
        start = clip.get("start", 0)
        end = clip.get("end", 0)

        # Skip unknown pitch types
        if pitch_type == "unknown":
            continue

        # Skip negative or invalid timestamps
        if start < 0 or end < 0 or end <= start:
            continue

        valid[clip_id] = clip

    return valid, len(data) - len(valid)


def filter_by_types(data: dict, pitch_types: list) -> dict:
    """Filter dataset to only include specified pitch types."""
    return {
        clip_id: clip
        for clip_id, clip in data.items()
        if clip.get("type", "unknown") in pitch_types
    }


def group_by_video(data: dict) -> dict:
    """Group clips by their source YouTube video."""
    by_video = defaultdict(list)
    for clip_id, clip in data.items():
        url = clip["url"]
        video_id = url.split("=")[-1]
        by_video[video_id].append((clip_id, clip))
    return dict(by_video)


def has_completed_download(output_path: Path) -> bool:
    """Return True if the output file looks like a completed download."""
    return output_path.exists() and output_path.stat().st_size > 0


def cleanup_partial_downloads(output_path: Path):
    """Remove stale partial files left from interrupted downloads."""
    partial_candidates = [
        output_path.with_suffix(".partial.mp4"),
        output_path.with_suffix(output_path.suffix + ".part"),
        output_path.with_name(output_path.name + ".part"),
    ]

    for partial_path in partial_candidates:
        if partial_path.exists():
            partial_path.unlink()


def download_clip(video_id: str, start: float, end: float, output_path: Path, timeout: int = 120) -> bool:
    """Download a specific clip segment directly from YouTube. Returns True on success."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if has_completed_download(output_path):
        return True
    if output_path.exists():
        output_path.unlink()

    cleanup_partial_downloads(output_path)

    url = f"https://www.youtube.com/watch?v={video_id}"
    temp_output_path = output_path.with_suffix(".partial.mp4")

    # yt-dlp --download-sections format: "*start-end"
    section = f"*{start}-{end}"

    cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]",
        "--download-sections", section,
        "--force-keyframes-at-cuts",
        "--merge-output-format", "mp4",
        "--no-warnings",
        "-o", str(temp_output_path),
        url
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout)
        if has_completed_download(temp_output_path):
            temp_output_path.replace(output_path)
            return True
        return has_completed_download(output_path)
    except subprocess.TimeoutExpired:
        log_error(f"Timeout downloading {output_path.name}")
        cleanup_partial_downloads(output_path)
        return False
    except subprocess.CalledProcessError as e:
        log_error(f"Failed to download {output_path.name}: {e.stderr}")
        cleanup_partial_downloads(output_path)
        return False


def log_error(message: str):
    """Append error message to log file."""
    with ERROR_LOG_LOCK:
        with open(ERROR_LOG, "a") as f:
            f.write(message + "\n")


def save_metadata(data: dict, output_dir: Path):
    """Save metadata CSV for all clips."""
    csv_path = output_dir / "metadata.csv"

    with open(csv_path, "w") as f:
        f.write("clip_id,pitch_type,speed,subset,labels\n")
        for clip_id, clip in sorted(data.items()):
            pitch_type = clip.get("type", "unknown")
            speed = clip.get("speed", "")
            subset = clip.get("subset", "")
            labels = "|".join(clip.get("labels", []))
            f.write(f"{clip_id},{pitch_type},{speed},{subset},{labels}\n")


def build_download_jobs(data: dict) -> list[tuple[str, dict, Path]]:
    """Build download jobs grouped by source video to improve request locality."""
    jobs = []
    for _, clips in sorted(group_by_video(data).items(), key=lambda item: item[0]):
        for clip_id, clip in clips:
            pitch_type = clip.get("type", "unknown")
            output_path = OUTPUT_DIR / pitch_type / f"{clip_id}.mp4"
            jobs.append((clip_id, clip, output_path))
    return jobs


def download_all_clips(jobs: list[tuple[str, dict, Path]], workers: int, timeout: int) -> tuple[int, int]:
    """Download clips sequentially or in parallel."""
    successful = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                download_clip,
                clip["url"].split("=")[-1],
                clip["start"],
                clip["end"],
                output_path,
                timeout,
            ): clip_id
            for clip_id, clip, output_path in jobs
        }

        try:
            for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading clips"):
                if future.result():
                    successful += 1
                else:
                    failed += 1
        except KeyboardInterrupt:
            print("\nInterrupted. Completed downloads were kept; in-flight partial files were not promoted.")
            for future in futures:
                future.cancel()
            raise

    return successful, failed


def main():
    parser = argparse.ArgumentParser(description="Download MLB pitch video clips")
    parser.add_argument("--limit", type=int, help="Limit number of clips (for testing)")
    parser.add_argument("--types", nargs="+", help="Only download specific pitch types")
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of concurrent downloads (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Per-clip download timeout in seconds (default: 120)",
    )
    args = parser.parse_args()

    # Check dependencies
    check_dependencies()

    # Load dataset
    print("Loading dataset...")
    data = load_dataset(DATA_JSON)
    print(f"Total clips in dataset: {len(data)}")

    # Filter out bad data (unknown pitch types, negative timestamps)
    data, removed = filter_valid_clips(data)
    print(f"Filtered out {removed} invalid clips (unknown type or bad timestamps)")
    print(f"Valid clips: {len(data)}")

    # Filter by pitch types if specified
    if args.types:
        data = filter_by_types(data, args.types)
        print(f"Filtered to {len(data)} clips ({', '.join(args.types)})")

    # Apply limit with stratified sampling
    if args.limit and args.limit < len(data):
        data = stratified_sample(data, args.limit)
        print(f"Sampled {len(data)} clips (stratified by pitch type)")

    # Show pitch type distribution
    type_counts = defaultdict(int)
    for clip in data.values():
        type_counts[clip.get("type", "unknown")] += 1
    print("\nPitch type distribution:")
    for ptype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {ptype}: {count}")

    jobs = build_download_jobs(data)
    unique_videos = len(group_by_video(data))

    # Process clips with bounded concurrency
    print(f"\nDownloading {len(jobs)} clips from {unique_videos} source videos...")
    print(f"Using {args.workers} concurrent download worker(s)")

    successful, failed = download_all_clips(jobs, args.workers, args.timeout)

    # Save metadata
    save_metadata(data, OUTPUT_DIR)

    # Summary
    print(f"\nComplete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Output: {OUTPUT_DIR}/")

    if failed > 0:
        print(f"  Errors logged to: {ERROR_LOG}")


if __name__ == "__main__":
    main()
