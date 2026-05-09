#!/usr/bin/env python3
"""
Export Stage B final debug contact sheets for visual QA.

Usage:
    python -m stage_b.export_debug_contacts
    python -m stage_b.export_debug_contacts --limit 50 --sort lowest-confidence
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from stage_b.paths import DEBUG_DIR, FINAL_EVENTS_CSV, FRAME_EXPORTS_CSV, SEQUENCES_DIR, ensure_stage_b_dirs


CONTACT_DIR = DEBUG_DIR / "contacts"
TIMELINE_THUMB_SIZE = (150, 150)
SEQUENCE_THUMB_SIZE = (110, 110)
PADDING = 10
TIMELINE_LABEL_HEIGHT = 42
SEQUENCE_LABEL_HEIGHT = 28
HEADER_HEIGHT = 98
SECTION_GAP = 18
BACKGROUND = (17, 21, 25)
CARD_FILL = (28, 36, 42)
TEXT = (238, 241, 245)
MUTED = (163, 172, 184)
NORMAL_BORDER = (74, 163, 223)
RELEASE_BORDER = (87, 167, 255)
CATCH_BORDER = (255, 153, 72)
SEQUENCE_FILL = (37, 65, 49)


def load_csv_rows(path: Path) -> list[dict]:
    """Load CSV rows from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV at {path}")
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


def load_final_events(path: Path) -> dict[str, dict]:
    """Load final Stage B events keyed by clip id."""
    if not path.exists():
        return {}
    rows = load_csv_rows(path)
    events = {}
    for row in rows:
        row["release_frame_idx"] = int(row["release_frame_idx"])
        row["catch_frame_idx"] = int(row["catch_frame_idx"])
        row["event_confidence"] = float(row["event_confidence"])
        row["weak_confidence"] = float(row["weak_confidence"]) if row.get("weak_confidence") else 0.0
        events[row["clip_id"]] = row
    return events


def sample_rows(rows: list[dict], max_frames: int) -> list[dict]:
    """Evenly sample rows for the timeline while keeping endpoints."""
    if len(rows) <= max_frames:
        return rows

    last_index = len(rows) - 1
    selected_indices = []
    seen = set()
    for output_index in range(max_frames):
        source_index = round(output_index * last_index / (max_frames - 1))
        if source_index in seen:
            continue
        seen.add(source_index)
        selected_indices.append(source_index)
    return [rows[index] for index in selected_indices]


def ensure_key_frames(rows: list[dict], selected_rows: list[dict], release_frame_idx: int, catch_frame_idx: int) -> list[dict]:
    """Force release and catch frames into the displayed timeline if present."""
    frame_lookup = {row["frame_idx"]: row for row in rows}
    required = []
    for frame_idx in (release_frame_idx, catch_frame_idx):
        if frame_idx in frame_lookup:
            required.append(frame_lookup[frame_idx])

    merged = {row["frame_idx"]: row for row in selected_rows}
    for row in required:
        merged[row["frame_idx"]] = row
    return [merged[frame_idx] for frame_idx in sorted(merged)]


def choose_clip_ids(
    grouped_rows: dict[str, list[dict]],
    final_events: dict[str, dict],
    sort_mode: str,
    limit: int | None,
) -> list[str]:
    """Choose clips to export in a useful review order."""
    clip_ids = [clip_id for clip_id in grouped_rows if clip_id in final_events]
    if sort_mode == "lowest-confidence":
        clip_ids.sort(key=lambda clip_id: final_events[clip_id]["event_confidence"])
    elif sort_mode == "longest-window":
        clip_ids.sort(
            key=lambda clip_id: final_events[clip_id]["catch_frame_idx"] - final_events[clip_id]["release_frame_idx"],
            reverse=True,
        )
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


def sequence_path_for_event(event: dict) -> Path:
    """Infer the saved sequence path from event metadata."""
    label = "offspeed" if event["pitch_type"] in {"slider", "curveball", "changeup", "sinker", "knucklecurve"} else "fastball"
    source_to_split = {"training": "train", "validation": "val", "testing": "test"}

    # Find file under any split if final-events CSV does not encode it.
    for split in ("train", "val", "test"):
        path = SEQUENCES_DIR / split / label / f"{event['clip_id']}.npy"
        if path.exists():
            return path
    return SEQUENCES_DIR / source_to_split.get("", "train") / label / f"{event['clip_id']}.npy"


def render_timeline(
    sheet: Image.Image,
    draw: ImageDraw.ImageDraw,
    rows: list[dict],
    event: dict,
    start_y: int,
    max_frames: int,
    columns: int,
    label_font,
    small_font,
) -> int:
    """Render the candidate timeline strip/grid and return the next y coordinate."""
    displayed_rows = ensure_key_frames(
        rows,
        sample_rows(rows, max_frames),
        release_frame_idx=event["release_frame_idx"],
        catch_frame_idx=event["catch_frame_idx"],
    )

    cell_width = TIMELINE_THUMB_SIZE[0] + PADDING
    cell_height = TIMELINE_THUMB_SIZE[1] + TIMELINE_LABEL_HEIGHT + PADDING
    for index, row in enumerate(displayed_rows):
        col = index % columns
        row_num = index // columns
        x = PADDING + col * cell_width
        y = start_y + row_num * cell_height

        frame_idx = row["frame_idx"]
        in_window = event["release_frame_idx"] <= frame_idx <= event["catch_frame_idx"]

        if in_window:
            draw.rectangle(
                [x - 3, y - 3, x + TIMELINE_THUMB_SIZE[0] + 3, y + TIMELINE_THUMB_SIZE[1] + TIMELINE_LABEL_HEIGHT - 3],
                fill=SEQUENCE_FILL,
            )
        else:
            draw.rectangle(
                [x - 3, y - 3, x + TIMELINE_THUMB_SIZE[0] + 3, y + TIMELINE_THUMB_SIZE[1] + TIMELINE_LABEL_HEIGHT - 3],
                fill=CARD_FILL,
            )

        image = Image.open(row["frame_path"]).convert("RGB")
        image.thumbnail(TIMELINE_THUMB_SIZE)
        thumb = Image.new("RGB", TIMELINE_THUMB_SIZE, (0, 0, 0))
        thumb_x = (TIMELINE_THUMB_SIZE[0] - image.width) // 2
        thumb_y = (TIMELINE_THUMB_SIZE[1] - image.height) // 2
        thumb.paste(image, (thumb_x, thumb_y))
        sheet.paste(thumb, (x, y))

        if frame_idx == event["release_frame_idx"]:
            border = RELEASE_BORDER
            label = "release"
        elif frame_idx == event["catch_frame_idx"]:
            border = CATCH_BORDER
            label = "catch"
        else:
            border = NORMAL_BORDER
            label = "window" if in_window else "candidate"
        draw.rectangle([x, y, x + TIMELINE_THUMB_SIZE[0] - 1, y + TIMELINE_THUMB_SIZE[1] - 1], outline=border, width=3)

        label_y = y + TIMELINE_THUMB_SIZE[1] + 6
        draw_text(draw, (x, label_y), f"frame {frame_idx}", label_font)
        draw_text(draw, (x, label_y + 18), label, small_font, fill=MUTED)

    rows_used = (len(displayed_rows) + columns - 1) // columns
    return start_y + rows_used * cell_height


def render_sequence_strip(
    sheet: Image.Image,
    draw: ImageDraw.ImageDraw,
    sequence: np.ndarray,
    start_y: int,
    label_font,
) -> int:
    """Render the final exported sequence and return the next y coordinate."""
    draw_text(draw, (PADDING, start_y), "Final Exported Sequence", label_font)
    y = start_y + 28
    cell_width = SEQUENCE_THUMB_SIZE[0] + PADDING
    for index, frame in enumerate(sequence):
        x = PADDING + index * cell_width
        thumb = Image.fromarray(np.clip(frame * 255.0, 0, 255).astype(np.uint8))
        thumb = thumb.resize(SEQUENCE_THUMB_SIZE)
        draw.rectangle([x - 3, y - 3, x + SEQUENCE_THUMB_SIZE[0] + 3, y + SEQUENCE_THUMB_SIZE[1] + SEQUENCE_LABEL_HEIGHT - 3], fill=CARD_FILL)
        sheet.paste(thumb, (x, y))
        draw.rectangle([x, y, x + SEQUENCE_THUMB_SIZE[0] - 1, y + SEQUENCE_THUMB_SIZE[1] - 1], outline=NORMAL_BORDER, width=2)
        draw_text(draw, (x, y + SEQUENCE_THUMB_SIZE[1] + 6), f"step {index + 1}", label_font, fill=MUTED)
    return y + SEQUENCE_THUMB_SIZE[1] + SEQUENCE_LABEL_HEIGHT


def render_contact_sheet(clip_id: str, rows: list[dict], event: dict, max_frames: int, columns: int) -> Image.Image:
    """Render one clip's final Stage B debug contact sheet."""
    title_font = load_font(22)
    label_font = load_font(14)
    small_font = load_font(12)

    sequence_path = sequence_path_for_event(event)
    if not sequence_path.exists():
        raise FileNotFoundError(
            f"Missing exported sequence for {clip_id} at {sequence_path}. Run stage_b.export_sequences first."
        )
    sequence = np.load(sequence_path)

    displayed_rows = ensure_key_frames(rows, sample_rows(rows, max_frames), event["release_frame_idx"], event["catch_frame_idx"])
    timeline_rows = (len(displayed_rows) + columns - 1) // columns
    timeline_height = timeline_rows * (TIMELINE_THUMB_SIZE[1] + TIMELINE_LABEL_HEIGHT + PADDING)
    sequence_height = 28 + SEQUENCE_THUMB_SIZE[1] + SEQUENCE_LABEL_HEIGHT

    width = max(
        PADDING + columns * (TIMELINE_THUMB_SIZE[0] + PADDING),
        PADDING + sequence.shape[0] * (SEQUENCE_THUMB_SIZE[0] + PADDING),
    )
    height = HEADER_HEIGHT + timeline_height + SECTION_GAP + sequence_height + PADDING

    sheet = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(sheet)

    first = rows[0]
    draw_text(draw, (PADDING, 12), f"{clip_id}  |  {event['pitch_type']}", title_font)
    draw_text(
        draw,
        (PADDING, 42),
        (
            f"release={event['release_frame_idx']}  |  catch={event['catch_frame_idx']}  |  "
            f"source={event['event_source']}  |  confidence={event['event_confidence']:.3f}"
        ),
        small_font,
        fill=MUTED,
    )
    draw_text(
        draw,
        (PADDING, 62),
        (
            f"candidate frames {first['segment_start_frame']}-{first['segment_end_frame']}  |  "
            f"crop={first['crop_left']},{first['crop_top']},{first['crop_right']},{first['crop_bottom']}"
        ),
        small_font,
        fill=MUTED,
    )

    next_y = render_timeline(
        sheet=sheet,
        draw=draw,
        rows=rows,
        event=event,
        start_y=HEADER_HEIGHT,
        max_frames=max_frames,
        columns=columns,
        label_font=label_font,
        small_font=small_font,
    )
    render_sequence_strip(
        sheet=sheet,
        draw=draw,
        sequence=sequence,
        start_y=next_y + SECTION_GAP,
        label_font=label_font,
    )
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Stage B final debug contact sheets")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of clips to export")
    parser.add_argument(
        "--sort",
        choices=["clip-id", "lowest-confidence", "longest-window"],
        default="lowest-confidence",
        help="Order for exported contact sheets",
    )
    parser.add_argument("--max-frames", type=int, default=48, help="Maximum candidate thumbnails per sheet")
    parser.add_argument("--columns", type=int, default=8, help="Number of candidate thumbnails per row")
    args = parser.parse_args()

    ensure_stage_b_dirs()
    CONTACT_DIR.mkdir(parents=True, exist_ok=True)

    grouped_rows = group_frame_rows(load_csv_rows(FRAME_EXPORTS_CSV))
    final_events = load_final_events(FINAL_EVENTS_CSV)
    clip_ids = choose_clip_ids(grouped_rows, final_events, args.sort, args.limit)

    exported = 0
    for clip_id in clip_ids:
        sheet = render_contact_sheet(
            clip_id=clip_id,
            rows=grouped_rows[clip_id],
            event=final_events[clip_id],
            max_frames=args.max_frames,
            columns=args.columns,
        )
        output_path = CONTACT_DIR / f"{clip_id}_stage_b_final.jpg"
        sheet.save(output_path, quality=92)
        exported += 1

    print(f"Exported {exported} Stage B final debug contact sheet(s) to: {CONTACT_DIR}")


if __name__ == "__main__":
    main()
