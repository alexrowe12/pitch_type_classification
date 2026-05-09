#!/usr/bin/env python3
"""Export compact Stage A and Stage B contact sheets for presentation."""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocess.paths import CLIPS_DIR, DATA_DIR
from stage_a.paths import CLIP_SEGMENTS_CSV, FRAME_PREDICTIONS_CSV
from stage_b.paths import FINAL_EVENTS_CSV, FRAME_EXPORTS_CSV, SEQUENCES_DIR


OUTPUT_DIR = DATA_DIR / "presentation_contacts"
STAGE_A_OUT = OUTPUT_DIR / "stage_a"
STAGE_B_OUT = OUTPUT_DIR / "stage_b"

BG = (17, 21, 25)
TEXT = (240, 243, 247)
MUTED = (165, 173, 184)
GREEN = (70, 195, 120)
BLUE = (86, 160, 235)
ORANGE = (255, 155, 78)
RED = (220, 75, 75)
CARD = (28, 36, 42)


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, text_font, fill=TEXT) -> None:
    x, y = xy
    draw.text((x + 1, y + 1), text, font=text_font, fill=(0, 0, 0))
    draw.text((x, y), text, font=text_font, fill=fill)


def read_video_frame(video_path: Path, frame_idx: int) -> Image.Image | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        return None
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def fit_thumb(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail(size)
    thumb = Image.new("RGB", size, (0, 0, 0))
    thumb.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return thumb


def evenly_sample(rows: list[dict], count: int) -> list[dict]:
    if len(rows) <= count:
        return rows
    last = len(rows) - 1
    indices = sorted({round(i * last / (count - 1)) for i in range(count)})
    return [rows[i] for i in indices]


def group_stage_a_predictions() -> dict[str, list[dict]]:
    rows = load_csv(FRAME_PREDICTIONS_CSV)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        row["frame_idx"] = int(row["frame_idx"])
        row["pitch_camera_probability"] = float(row["pitch_camera_probability"])
        grouped[row["clip_id"]].append(row)
    for clip_rows in grouped.values():
        clip_rows.sort(key=lambda row: row["frame_idx"])
    return grouped


def load_stage_a_segments() -> list[dict]:
    rows = load_csv(CLIP_SEGMENTS_CSV)
    for row in rows:
        row["segment_start_frame"] = int(row["segment_start_frame"])
        row["segment_end_frame"] = int(row["segment_end_frame"])
        row["segment_mean_probability"] = float(row["segment_mean_probability"])
    rows.sort(key=lambda row: row["segment_mean_probability"], reverse=True)
    return rows


def export_stage_a(limit: int, max_frames: int, columns: int, threshold: float) -> int:
    STAGE_A_OUT.mkdir(parents=True, exist_ok=True)
    predictions = group_stage_a_predictions()
    segments = load_stage_a_segments()
    title_font = font(22)
    label_font = font(13)
    small_font = font(11)
    thumb_size = (192, 108)
    pad = 10
    header_h = 74
    label_h = 38
    exported = 0

    for segment in segments:
        if exported >= limit:
            break
        clip_id = segment["clip_id"]
        rows = predictions.get(clip_id, [])
        if not rows:
            continue
        video_path = CLIPS_DIR / segment["pitch_type"] / f"{clip_id}.mp4"
        if not video_path.exists():
            continue

        sampled_rows = evenly_sample(rows, max_frames)
        grid_rows = (len(sampled_rows) + columns - 1) // columns
        width = pad + columns * (thumb_size[0] + pad)
        height = header_h + grid_rows * (thumb_size[1] + label_h + pad) + pad
        sheet = Image.new("RGB", (width, height), BG)
        draw = ImageDraw.Draw(sheet)

        draw_label(draw, (pad, 12), f"Stage A Shot Detection | {clip_id} | {segment['pitch_type']}", title_font)
        draw_label(
            draw,
            (pad, 42),
            f"green = pitch camera above {threshold:.2f} | selected segment {segment['segment_start_frame']}-{segment['segment_end_frame']} | mean p={segment['segment_mean_probability']:.3f}",
            small_font,
            MUTED,
        )

        for idx, row in enumerate(sampled_rows):
            image = read_video_frame(video_path, row["frame_idx"])
            if image is None:
                continue
            x = pad + (idx % columns) * (thumb_size[0] + pad)
            y = header_h + (idx // columns) * (thumb_size[1] + label_h + pad)
            prob = row["pitch_camera_probability"]
            frame_idx = row["frame_idx"]
            in_segment = segment["segment_start_frame"] <= frame_idx <= segment["segment_end_frame"]
            border = GREEN if prob >= threshold else RED
            sheet.paste(fit_thumb(image, thumb_size), (x, y))
            draw.rectangle([x, y, x + thumb_size[0] - 1, y + thumb_size[1] - 1], outline=border, width=4 if in_segment else 2)
            draw_label(draw, (x, y + thumb_size[1] + 5), f"frame {frame_idx}", label_font)
            draw_label(draw, (x, y + thumb_size[1] + 22), f"pitch p={prob:.3f}", small_font, MUTED)

        output_path = STAGE_A_OUT / f"{exported + 1:02d}_{clip_id}_stage_a.jpg"
        sheet.save(output_path, quality=92)
        exported += 1
    return exported


def load_stage_b_rows() -> tuple[dict[str, list[dict]], list[dict]]:
    frame_rows = load_csv(FRAME_EXPORTS_CSV)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in frame_rows:
        row["frame_idx"] = int(row["frame_idx"])
        grouped[row["clip_id"]].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: row["frame_idx"])

    events = load_csv(FINAL_EVENTS_CSV)
    for row in events:
        row["release_frame_idx"] = int(row["release_frame_idx"])
        row["catch_frame_idx"] = int(row["catch_frame_idx"])
        row["event_confidence"] = float(row["event_confidence"])
    events.sort(key=lambda row: row["event_confidence"], reverse=True)
    return grouped, events


def sequence_path(event: dict) -> Path | None:
    binary_label = "fastball" if event["pitch_type"] == "fastball" else "offspeed"
    for split in ("train", "val", "test"):
        path = SEQUENCES_DIR / split / binary_label / f"{event['clip_id']}.npy"
        if path.exists():
            return path
    return None


def select_stage_b_timeline(rows: list[dict], release: int, catch: int, count: int) -> list[dict]:
    before = [row for row in rows if row["frame_idx"] < release]
    window = [row for row in rows if release <= row["frame_idx"] <= catch]
    after = [row for row in rows if row["frame_idx"] > catch]
    selected = evenly_sample(before[-6:], min(4, len(before[-6:])))
    selected.extend(evenly_sample(window, max(4, count - len(selected) - 2)))
    selected.extend(evenly_sample(after[:6], min(2, len(after[:6]))))
    by_frame = {row["frame_idx"]: row for row in selected}
    for target in (release, catch):
        nearest = min(rows, key=lambda row: abs(row["frame_idx"] - target))
        by_frame[nearest["frame_idx"]] = nearest
    return [by_frame[idx] for idx in sorted(by_frame)]


def export_stage_b(limit: int, max_frames: int, columns: int) -> int:
    STAGE_B_OUT.mkdir(parents=True, exist_ok=True)
    grouped, events = load_stage_b_rows()
    title_font = font(22)
    label_font = font(13)
    small_font = font(11)
    timeline_size = (150, 150)
    sequence_size = (96, 96)
    pad = 10
    header_h = 78
    label_h = 38
    seq_header_h = 34
    exported = 0

    for event in events:
        if exported >= limit:
            break
        clip_id = event["clip_id"]
        rows = grouped.get(clip_id, [])
        path = sequence_path(event)
        if not rows or path is None:
            continue

        timeline = select_stage_b_timeline(rows, event["release_frame_idx"], event["catch_frame_idx"], max_frames)
        sequence = np.load(path)
        timeline_rows = (len(timeline) + columns - 1) // columns
        width = pad + max(columns * (timeline_size[0] + pad), len(sequence) * (sequence_size[0] + pad))
        timeline_h = timeline_rows * (timeline_size[1] + label_h + pad)
        height = header_h + timeline_h + seq_header_h + sequence_size[1] + label_h + pad
        sheet = Image.new("RGB", (width, height), BG)
        draw = ImageDraw.Draw(sheet)

        draw_label(draw, (pad, 12), f"Stage B Release/Catch | {clip_id} | {event['pitch_type']}", title_font)
        draw_label(
            draw,
            (pad, 42),
            f"release frame {event['release_frame_idx']} | catch frame {event['catch_frame_idx']} | final 12-frame sequence below",
            small_font,
            MUTED,
        )

        for idx, row in enumerate(timeline):
            x = pad + (idx % columns) * (timeline_size[0] + pad)
            y = header_h + (idx // columns) * (timeline_size[1] + label_h + pad)
            frame_idx = row["frame_idx"]
            image = Image.open(row["frame_path"])
            in_window = event["release_frame_idx"] <= frame_idx <= event["catch_frame_idx"]
            draw.rectangle([x - 3, y - 3, x + timeline_size[0] + 3, y + timeline_size[1] + label_h - 3], fill=CARD)
            sheet.paste(fit_thumb(image, timeline_size), (x, y))
            if frame_idx == event["release_frame_idx"]:
                border, label = BLUE, "release"
            elif frame_idx == event["catch_frame_idx"]:
                border, label = ORANGE, "catch"
            else:
                border, label = GREEN if in_window else RED, "pitch window" if in_window else "context"
            draw.rectangle([x, y, x + timeline_size[0] - 1, y + timeline_size[1] - 1], outline=border, width=3)
            draw_label(draw, (x, y + timeline_size[1] + 5), f"frame {frame_idx}", label_font)
            draw_label(draw, (x, y + timeline_size[1] + 22), label, small_font, MUTED)

        seq_y = header_h + timeline_h + 10
        draw_label(draw, (pad, seq_y), "Final model input sequence", label_font)
        seq_y += seq_header_h
        for idx, frame in enumerate(sequence):
            x = pad + idx * (sequence_size[0] + pad)
            image = Image.fromarray(np.clip(frame * 255.0, 0, 255).astype(np.uint8))
            sheet.paste(fit_thumb(image, sequence_size), (x, seq_y))
            draw.rectangle([x, seq_y, x + sequence_size[0] - 1, seq_y + sequence_size[1] - 1], outline=GREEN, width=2)
            draw_label(draw, (x, seq_y + sequence_size[1] + 5), f"step {idx + 1}", small_font, MUTED)

        output_path = STAGE_B_OUT / f"{exported + 1:02d}_{clip_id}_stage_b.jpg"
        sheet.save(output_path, quality=92)
        exported += 1
    return exported


def main() -> None:
    parser = argparse.ArgumentParser(description="Export compact presentation contact sheets")
    parser.add_argument("--stage-a-limit", type=int, default=6)
    parser.add_argument("--stage-b-limit", type=int, default=6)
    parser.add_argument("--stage-a-max-frames", type=int, default=24)
    parser.add_argument("--stage-b-max-frames", type=int, default=14)
    parser.add_argument("--columns", type=int, default=6)
    parser.add_argument("--stage-a-threshold", type=float, default=0.98)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stage_a_count = export_stage_a(
        limit=args.stage_a_limit,
        max_frames=args.stage_a_max_frames,
        columns=args.columns,
        threshold=args.stage_a_threshold,
    )
    stage_b_count = export_stage_b(
        limit=args.stage_b_limit,
        max_frames=args.stage_b_max_frames,
        columns=args.columns,
    )
    print(f"Exported {stage_a_count} Stage A sheet(s) to: {STAGE_A_OUT}")
    print(f"Exported {stage_b_count} Stage B sheet(s) to: {STAGE_B_OUT}")


if __name__ == "__main__":
    main()
