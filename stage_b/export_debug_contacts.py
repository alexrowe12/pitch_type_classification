#!/usr/bin/env python3
"""
Export Stage B candidate contact sheets for visual QA.

Usage:
    python -m stage_b.export_debug_contacts
    python -m stage_b.export_debug_contacts --limit 50 --sort longest-segment
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from stage_b.paths import DEBUG_DIR, FRAME_EXPORTS_CSV, ensure_stage_b_dirs


CONTACT_DIR = DEBUG_DIR / "contacts"
THUMB_SIZE = (160, 160)
PADDING = 10
HEADER_HEIGHT = 86
LABEL_HEIGHT = 40
BACKGROUND = (17, 21, 25)
CARD_FILL = (28, 36, 42)
TEXT = (238, 241, 245)
MUTED = (163, 172, 184)
BORDER = (74, 163, 223)


def load_csv_rows(path: Path) -> list[dict]:
    """Load CSV rows from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Missing Stage B frame exports at {path}. Run stage_b.export_candidates first.")
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def group_frame_rows(rows: list[dict]) -> dict[str, list[dict]]:
    """Group candidate frame rows by clip."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        row["frame_idx"] = int(row["frame_idx"])
        row["segment_start_frame"] = int(row["segment_start_frame"])
        row["segment_end_frame"] = int(row["segment_end_frame"])
        row["segment_mean_probability"] = float(row["segment_mean_probability"])
        grouped[row["clip_id"]].append(row)
    for clip_rows in grouped.values():
        clip_rows.sort(key=lambda row: row["frame_idx"])
    return grouped


def sample_rows(rows: list[dict], max_frames: int) -> list[dict]:
    """Evenly sample rows when there are too many frames for one contact sheet."""
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


def choose_clip_ids(grouped_rows: dict[str, list[dict]], sort_mode: str, limit: int | None) -> list[str]:
    """Choose clips to export in a useful review order."""
    clip_ids = list(grouped_rows)
    if sort_mode == "longest-segment":
        clip_ids.sort(key=lambda clip_id: len(grouped_rows[clip_id]), reverse=True)
    elif sort_mode == "shortest-segment":
        clip_ids.sort(key=lambda clip_id: len(grouped_rows[clip_id]))
    else:
        clip_ids.sort()

    if limit is not None:
        clip_ids = clip_ids[:limit]
    return clip_ids


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
    """Draw text with a small shadow for readability."""
    x, y = xy
    draw.text((x + 1, y + 1), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=fill)


def render_contact_sheet(clip_id: str, rows: list[dict], max_frames: int, columns: int) -> Image.Image:
    """Render one clip's candidate frames as a contact sheet."""
    selected_rows = sample_rows(rows, max_frames)
    first = rows[0]
    title_font = load_font(22)
    label_font = load_font(14)
    small_font = load_font(12)

    row_count = (len(selected_rows) + columns - 1) // columns
    cell_width = THUMB_SIZE[0] + PADDING
    cell_height = THUMB_SIZE[1] + LABEL_HEIGHT + PADDING
    width = PADDING + columns * cell_width
    height = HEADER_HEIGHT + row_count * cell_height + PADDING

    sheet = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(sheet)

    start = first["segment_start_frame"]
    end = first["segment_end_frame"]
    mean_prob = first["segment_mean_probability"]
    pitch_type = first["pitch_type"]
    crop = (
        first["crop_left"],
        first["crop_top"],
        first["crop_right"],
        first["crop_bottom"],
    )

    draw_text(draw, (PADDING, 12), f"{clip_id}  |  {pitch_type}", title_font)
    draw_text(
        draw,
        (PADDING, 42),
        f"candidate frames {start}-{end}  |  exported={len(rows)}  |  Stage A mean p={mean_prob:.3f}",
        small_font,
        fill=MUTED,
    )
    draw_text(draw, (PADDING, 62), f"crop left/top/right/bottom={crop}", small_font, fill=MUTED)

    for index, row in enumerate(selected_rows):
        col = index % columns
        row_num = index // columns
        x = PADDING + col * cell_width
        y = HEADER_HEIGHT + row_num * cell_height

        draw.rectangle(
            [x - 3, y - 3, x + THUMB_SIZE[0] + 3, y + THUMB_SIZE[1] + LABEL_HEIGHT - 3],
            fill=CARD_FILL,
        )

        image = Image.open(row["frame_path"]).convert("RGB")
        image.thumbnail(THUMB_SIZE)
        thumb = Image.new("RGB", THUMB_SIZE, (0, 0, 0))
        thumb_x = (THUMB_SIZE[0] - image.width) // 2
        thumb_y = (THUMB_SIZE[1] - image.height) // 2
        thumb.paste(image, (thumb_x, thumb_y))
        sheet.paste(thumb, (x, y))

        draw.rectangle([x, y, x + THUMB_SIZE[0] - 1, y + THUMB_SIZE[1] - 1], outline=BORDER, width=2)
        label_y = y + THUMB_SIZE[1] + 6
        draw_text(draw, (x, label_y), f"frame {row['frame_idx']}", label_font)
        draw_text(draw, (x, label_y + 18), f"t={float(row['timestamp_sec']):.3f}s", small_font, fill=MUTED)

    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Stage B candidate contact sheets")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of clips to export")
    parser.add_argument(
        "--sort",
        choices=["clip-id", "longest-segment", "shortest-segment"],
        default="clip-id",
        help="Order for exported contact sheets",
    )
    parser.add_argument("--max-frames", type=int, default=48, help="Maximum thumbnails per contact sheet")
    parser.add_argument("--columns", type=int, default=8, help="Number of thumbnails per row")
    args = parser.parse_args()

    ensure_stage_b_dirs()
    CONTACT_DIR.mkdir(parents=True, exist_ok=True)

    grouped_rows = group_frame_rows(load_csv_rows(FRAME_EXPORTS_CSV))
    clip_ids = choose_clip_ids(grouped_rows, args.sort, args.limit)

    exported = 0
    for clip_id in clip_ids:
        sheet = render_contact_sheet(
            clip_id=clip_id,
            rows=grouped_rows[clip_id],
            max_frames=args.max_frames,
            columns=args.columns,
        )
        output_path = CONTACT_DIR / f"{clip_id}_stage_b_candidates.jpg"
        sheet.save(output_path, quality=92)
        exported += 1

    print(f"Exported {exported} Stage B candidate contact sheet(s) to: {CONTACT_DIR}")


if __name__ == "__main__":
    main()
