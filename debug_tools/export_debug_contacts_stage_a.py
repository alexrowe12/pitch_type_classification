#!/usr/bin/env python3
"""
Export Stage A contact sheets for quick visual inspection.

Usage:
    python -m stage_a.export_debug_contacts
    python -m stage_a.export_debug_contacts --limit 100 --sort lowest-confidence
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from stage_a.paths import (
    CLIP_SEGMENTS_CSV,
    DEBUG_DIR,
    FRAME_PREDICTIONS_CSV,
    ensure_stage_a_dirs,
)


CONTACT_DIR = DEBUG_DIR / "contacts"
THUMB_SIZE = (224, 126)
PADDING = 12
HEADER_HEIGHT = 76
LABEL_HEIGHT = 46
BACKGROUND = (18, 22, 28)
TEXT = (238, 241, 245)
MUTED = (166, 173, 184)
PITCH_BORDER = (64, 190, 110)
NON_PITCH_BORDER = (220, 80, 80)
SEGMENT_FILL = (32, 74, 50)


def load_csv_rows(path: Path) -> list[dict]:
    """Load CSV rows from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def group_predictions(rows: list[dict]) -> dict[str, list[dict]]:
    """Group frame predictions by clip."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        row["frame_idx"] = int(row["frame_idx"])
        row["pitch_camera_probability"] = float(row["pitch_camera_probability"])
        grouped[row["clip_id"]].append(row)
    for clip_rows in grouped.values():
        clip_rows.sort(key=lambda row: row["frame_idx"])
    return grouped


def load_segments(rows: list[dict]) -> dict[str, dict]:
    """Load clip segment rows keyed by clip id."""
    segments = {}
    for row in rows:
        row["segment_start_frame"] = int(row["segment_start_frame"])
        row["segment_end_frame"] = int(row["segment_end_frame"])
        row["segment_mean_probability"] = float(row["segment_mean_probability"])
        row["segment_num_sampled_frames"] = int(row["segment_num_sampled_frames"])
        segments[row["clip_id"]] = row
    return segments


def choose_clip_ids(segments: dict[str, dict], sort_mode: str, limit: int | None) -> list[str]:
    """Choose clips to export in a useful review order."""
    clip_ids = list(segments)
    if sort_mode == "lowest-confidence":
        clip_ids.sort(key=lambda clip_id: segments[clip_id]["segment_mean_probability"])
    elif sort_mode == "shortest-segment":
        clip_ids.sort(key=lambda clip_id: segments[clip_id]["segment_num_sampled_frames"])
    else:
        clip_ids.sort()

    if limit is not None:
        clip_ids = clip_ids[:limit]
    return clip_ids


def sample_rows(rows: list[dict], max_frames: int) -> list[dict]:
    """Evenly sample rows when a clip has more frames than fit cleanly on one sheet."""
    if len(rows) <= max_frames:
        return rows

    last_index = len(rows) - 1
    sampled = []
    seen_indices = set()
    for output_index in range(max_frames):
        source_index = round(output_index * last_index / (max_frames - 1))
        if source_index in seen_indices:
            continue
        seen_indices.add(source_index)
        sampled.append(rows[source_index])
    return sampled


def load_font(size: int) -> ImageFont.ImageFont:
    """Load a readable font with a PIL default fallback."""
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font, fill=TEXT) -> None:
    """Draw text with a tiny shadow for readability."""
    x, y = xy
    draw.text((x + 1, y + 1), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=fill)


def render_contact_sheet(
    clip_id: str,
    rows: list[dict],
    segment: dict,
    max_frames: int,
    columns: int,
    pitch_threshold: float,
) -> Image.Image:
    """Render one clip contact sheet."""
    selected_rows = sample_rows(rows, max_frames)
    title_font = load_font(24)
    label_font = load_font(15)
    small_font = load_font(13)

    rows_count = (len(selected_rows) + columns - 1) // columns
    width = PADDING + columns * (THUMB_SIZE[0] + PADDING)
    height = HEADER_HEIGHT + rows_count * (THUMB_SIZE[1] + LABEL_HEIGHT + PADDING) + PADDING

    sheet = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(sheet)

    start = segment["segment_start_frame"]
    end = segment["segment_end_frame"]
    mean_prob = segment["segment_mean_probability"]
    segment_len = segment["segment_num_sampled_frames"]
    pitch_type = segment["pitch_type"]

    draw_text(draw, (PADDING, 12), f"{clip_id}  |  {pitch_type}", title_font)
    subtitle = (
        f"selected segment: frames {start}-{end}  |  "
        f"mean p={mean_prob:.3f}  |  sampled frames={segment_len}  |  "
        f"green threshold={pitch_threshold:.3f}"
    )
    draw_text(draw, (PADDING, 44), subtitle, small_font, fill=MUTED)

    for index, row in enumerate(selected_rows):
        col = index % columns
        row_num = index // columns
        x = PADDING + col * (THUMB_SIZE[0] + PADDING)
        y = HEADER_HEIGHT + row_num * (THUMB_SIZE[1] + LABEL_HEIGHT + PADDING)

        image = Image.open(row["frame_path"]).convert("RGB")
        image.thumbnail(THUMB_SIZE)
        thumb = Image.new("RGB", THUMB_SIZE, (0, 0, 0))
        thumb_x = (THUMB_SIZE[0] - image.width) // 2
        thumb_y = (THUMB_SIZE[1] - image.height) // 2
        thumb.paste(image, (thumb_x, thumb_y))

        frame_idx = row["frame_idx"]
        prob = row["pitch_camera_probability"]
        in_segment = start <= frame_idx <= end
        predicted_pitch = prob >= pitch_threshold

        if in_segment:
            draw.rectangle(
                [x - 4, y - 4, x + THUMB_SIZE[0] + 4, y + THUMB_SIZE[1] + LABEL_HEIGHT - 4],
                fill=SEGMENT_FILL,
            )
        sheet.paste(thumb, (x, y))

        border = PITCH_BORDER if predicted_pitch else NON_PITCH_BORDER
        border_width = 5 if in_segment else 2
        for offset in range(border_width):
            draw.rectangle(
                [
                    x - offset,
                    y - offset,
                    x + THUMB_SIZE[0] - 1 + offset,
                    y + THUMB_SIZE[1] - 1 + offset,
                ],
                outline=border,
            )

        label_y = y + THUMB_SIZE[1] + 6
        draw_text(draw, (x, label_y), f"frame {frame_idx}", label_font)
        draw_text(draw, (x, label_y + 19), f"pitch prob {prob:.3f}", small_font, fill=MUTED)

    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Stage A contact-sheet debug images")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of clips to export")
    parser.add_argument(
        "--sort",
        choices=["clip-id", "lowest-confidence", "shortest-segment"],
        default="lowest-confidence",
        help="Order for exported contact sheets",
    )
    parser.add_argument("--max-frames", type=int, default=36, help="Maximum frames per contact sheet")
    parser.add_argument("--columns", type=int, default=6, help="Number of thumbnails per row")
    parser.add_argument(
        "--pitch-threshold",
        type=float,
        default=0.98,
        help="Probability threshold for drawing pitch-camera frames with green borders",
    )
    args = parser.parse_args()

    ensure_stage_a_dirs()
    CONTACT_DIR.mkdir(parents=True, exist_ok=True)

    prediction_rows = load_csv_rows(FRAME_PREDICTIONS_CSV)
    segment_rows = load_csv_rows(CLIP_SEGMENTS_CSV)
    predictions_by_clip = group_predictions(prediction_rows)
    segments_by_clip = load_segments(segment_rows)
    clip_ids = choose_clip_ids(segments_by_clip, args.sort, args.limit)

    exported = 0
    for clip_id in clip_ids:
        rows = predictions_by_clip.get(clip_id)
        if not rows:
            continue
        sheet = render_contact_sheet(
            clip_id=clip_id,
            rows=rows,
            segment=segments_by_clip[clip_id],
            max_frames=args.max_frames,
            columns=args.columns,
            pitch_threshold=args.pitch_threshold,
        )
        output_path = CONTACT_DIR / f"{clip_id}_stage_a_contact.jpg"
        sheet.save(output_path, quality=92)
        exported += 1

    print(f"Exported {exported} contact sheet(s) to: {CONTACT_DIR}")


if __name__ == "__main__":
    main()
