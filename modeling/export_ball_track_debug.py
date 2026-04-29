#!/usr/bin/env python3
"""
Export debug sheets for candidate baseball trajectory detection.

Usage:
    python -m modeling.export_ball_track_debug --limit 50
    python -m modeling.export_ball_track_debug --split val --limit 20
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from modeling.audit_dataset import LABELS, SPLITS
from modeling.export_variant_debug import draw_text, load_font
from modeling.paths import DEBUG_DIR, VARIANTS_DIR, ensure_modeling_dirs


CONTACT_DIR = DEBUG_DIR / "ball_track_candidates"
THUMB_SIZE = (150, 150)
PADDING = 10
HEADER_HEIGHT = 82
ROW_LABEL_WIDTH = 112
LABEL_HEIGHT = 38
BACKGROUND = (18, 22, 26)
CARD_FILL = (30, 37, 43)
TEXT = (238, 241, 245)
MUTED = (161, 170, 181)
CANDIDATE = (255, 194, 80)
SELECTED = (255, 66, 54)
DIFF_COLOR = (210, 210, 210)


@dataclass(frozen=True)
class Candidate:
    """One candidate ball blob in one sequence frame."""

    frame_index: int
    x: float
    y: float
    area: int
    width: int
    height: int
    motion_score: float
    color_score: float
    score: float


def list_rgb_sequences(variant_root: Path, split: str | None, limit: int | None) -> list[Path]:
    """Return RGB sequence paths in split/label order."""
    splits = [split] if split else list(SPLITS)
    paths = []
    for split_name in splits:
        for label in LABELS:
            paths.extend(sorted((variant_root / "rgb" / split_name / label).glob("*.npy")))
    if limit is not None:
        paths = paths[:limit]
    return paths


def grayscale(sequence: np.ndarray) -> np.ndarray:
    """Convert RGB sequence to grayscale."""
    return (
        0.299 * sequence[..., 0]
        + 0.587 * sequence[..., 1]
        + 0.114 * sequence[..., 2]
    ).astype(np.float32)


def temporal_diff(sequence: np.ndarray) -> np.ndarray:
    """Return normalized grayscale frame difference."""
    gray = grayscale(sequence)
    diff = np.zeros_like(gray, dtype=np.float32)
    diff[1:] = np.abs(gray[1:] - gray[:-1])
    max_value = float(diff.max())
    if max_value > 0:
        diff /= max_value
    return diff


def baseball_color_score(sequence: np.ndarray) -> np.ndarray:
    """Return a soft score for bright, low-saturation, warm white pixels."""
    red = sequence[..., 0]
    green = sequence[..., 1]
    blue = sequence[..., 2]
    max_channel = np.max(sequence, axis=-1)
    min_channel = np.min(sequence, axis=-1)
    saturation = (max_channel - min_channel) / np.maximum(max_channel, 1e-6)

    brightness = np.clip((max_channel - 0.48) / 0.35, 0.0, 1.0)
    low_saturation = np.clip((0.60 - saturation) / 0.60, 0.0, 1.0)
    warm_white = np.clip(((red + green) * 0.5 - blue + 0.18) / 0.40, 0.0, 1.0)
    return (brightness * low_saturation * warm_white).astype(np.float32)


def component_candidates(
    sequence: np.ndarray,
    diff: np.ndarray,
    color_score: np.ndarray,
    min_area: int,
    max_area: int,
    min_motion: float,
    min_color: float,
    min_y: float,
    max_y: float,
    max_candidates_per_frame: int,
) -> dict[int, list[Candidate]]:
    """Find connected-component candidates in every frame."""
    height, width = diff.shape[1:]
    y_min_px = int(round(height * min_y))
    y_max_px = int(round(height * max_y))
    kernel = np.ones((2, 2), dtype=np.uint8)
    candidates_by_frame: dict[int, list[Candidate]] = {}

    for frame_index in range(sequence.shape[0]):
        frame_diff = diff[frame_index]
        frame_color = color_score[frame_index]
        combined = frame_diff * frame_color
        adaptive_floor = float(np.percentile(combined, 99.35))
        threshold = max(min_motion * min_color, adaptive_floor)
        mask = ((frame_diff >= min_motion) & (frame_color >= min_color) & (combined >= threshold)).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        frame_candidates = []
        for label_index in range(1, num_labels):
            x, y, w, h, area = stats[label_index]
            if area < min_area or area > max_area:
                continue
            cx, cy = centroids[label_index]
            if cy < y_min_px or cy > y_max_px:
                continue
            if max(w, h) > 18:
                continue

            component_mask = labels == label_index
            motion_score = float(frame_diff[component_mask].mean())
            component_color = float(frame_color[component_mask].mean())
            area_score = float(np.exp(-abs(area - 5) / 10.0))
            score = (motion_score * 0.55) + (component_color * 0.35) + (area_score * 0.10)
            frame_candidates.append(
                Candidate(
                    frame_index=frame_index,
                    x=float(cx),
                    y=float(cy),
                    area=int(area),
                    width=int(w),
                    height=int(h),
                    motion_score=motion_score,
                    color_score=component_color,
                    score=score,
                )
            )

        frame_candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        candidates_by_frame[frame_index] = frame_candidates[:max_candidates_per_frame]
    return candidates_by_frame


def transition_score(previous: Candidate, current: Candidate) -> float:
    """Score the plausibility of connecting two candidates."""
    gap = current.frame_index - previous.frame_index
    if gap <= 0 or gap > 3:
        return -999.0

    dx = current.x - previous.x
    dy = current.y - previous.y
    distance = float(np.hypot(dx, dy))
    expected = 20.0 * gap
    distance_penalty = abs(distance - expected) / 45.0
    backward_penalty = 0.40 if dx < -6 else 0.0
    stationary_penalty = 0.25 if distance < 5 else 0.0
    gap_penalty = 0.10 * (gap - 1)
    return 0.25 - distance_penalty - backward_penalty - stationary_penalty - gap_penalty


def link_candidates(candidates_by_frame: dict[int, list[Candidate]]) -> list[Candidate]:
    """Link candidates into the best trajectory using dynamic programming."""
    all_candidates = [
        candidate
        for frame_candidates in candidates_by_frame.values()
        for candidate in frame_candidates
    ]
    if not all_candidates:
        return []

    ordered = sorted(all_candidates, key=lambda candidate: (candidate.frame_index, candidate.x))
    best_score: dict[Candidate, float] = {}
    previous_node: dict[Candidate, Candidate | None] = {}

    for candidate in ordered:
        best_score[candidate] = candidate.score
        previous_node[candidate] = None
        for previous in ordered:
            if previous.frame_index >= candidate.frame_index:
                break
            transition = transition_score(previous, candidate)
            if transition < -100:
                continue
            candidate_score = best_score[previous] + candidate.score + transition
            if candidate_score > best_score[candidate]:
                best_score[candidate] = candidate_score
                previous_node[candidate] = previous

    end = max(ordered, key=lambda candidate: best_score[candidate])
    path = []
    current: Candidate | None = end
    while current is not None:
        path.append(current)
        current = previous_node[current]
    path.reverse()

    if len(path) < 2:
        return []
    return path


def frame_to_image(frame: np.ndarray) -> Image.Image:
    """Convert normalized RGB frame to a thumbnail."""
    array = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB").resize(THUMB_SIZE)


def diff_to_image(frame: np.ndarray) -> Image.Image:
    """Convert normalized diff frame to a thumbnail."""
    array = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="L").convert("RGB").resize(THUMB_SIZE)


def scale_point(candidate: Candidate, source_size: int) -> tuple[int, int]:
    """Scale candidate center from source frame coordinates to thumbnail coordinates."""
    scale = THUMB_SIZE[0] / source_size
    return int(round(candidate.x * scale)), int(round(candidate.y * scale))


def draw_candidates(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    candidates: list[Candidate],
    selected: Candidate | None,
    source_size: int,
) -> None:
    """Draw candidate boxes and selected point on one thumbnail."""
    for candidate in candidates:
        cx, cy = scale_point(candidate, source_size)
        radius = 3
        color = SELECTED if candidate == selected else CANDIDATE
        width = 3 if candidate == selected else 1
        draw.ellipse(
            [x + cx - radius, y + cy - radius, x + cx + radius, y + cy + radius],
            outline=color,
            width=width,
        )


def render_sheet(
    rgb_path: Path,
    variant_root: Path,
    sequence: np.ndarray,
    diff: np.ndarray,
    candidates: dict[int, list[Candidate]],
    path: list[Candidate],
) -> Image.Image:
    """Render one ball-track debug sheet."""
    relative = rgb_path.relative_to(variant_root / "rgb")
    split, label, filename = relative.parts
    clip_id = Path(filename).stem
    columns = sequence.shape[0]
    source_size = sequence.shape[1]
    cell_width = THUMB_SIZE[0] + PADDING
    width = PADDING * 2 + ROW_LABEL_WIDTH + columns * cell_width
    row_height = THUMB_SIZE[1] + LABEL_HEIGHT + PADDING
    height = HEADER_HEIGHT + 2 * row_height + PADDING
    sheet = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(sheet)

    title_font = load_font(22)
    label_font = load_font(13)
    small_font = load_font(11)
    selected_by_frame = {candidate.frame_index: candidate for candidate in path}
    candidate_count = sum(len(items) for items in candidates.values())

    draw_text(draw, (PADDING, PADDING), f"{clip_id} | {split} | {label}", title_font)
    draw_text(
        draw,
        (PADDING, PADDING + 30),
        f"candidates={candidate_count} selected_path={len(path)}",
        label_font,
        fill=MUTED,
    )

    for row_name, row_index in (("RGB+Cands", 0), ("Diff+Cands", 1)):
        y = HEADER_HEIGHT + row_index * row_height
        draw_text(draw, (PADDING, y + THUMB_SIZE[1] // 2 - 6), row_name, label_font)
        for frame_index in range(columns):
            x = PADDING + ROW_LABEL_WIDTH + frame_index * cell_width
            thumb = frame_to_image(sequence[frame_index]) if row_index == 0 else diff_to_image(diff[frame_index])
            draw.rectangle([x - 3, y - 3, x + THUMB_SIZE[0] + 3, y + THUMB_SIZE[1] + LABEL_HEIGHT - 3], fill=CARD_FILL)
            sheet.paste(thumb, (x, y))
            selected = selected_by_frame.get(frame_index)
            draw_candidates(draw, x, y, candidates.get(frame_index, []), selected, source_size=source_size)
            draw.rectangle([x, y, x + THUMB_SIZE[0] - 1, y + THUMB_SIZE[1] - 1], outline=DIFF_COLOR, width=1)
            draw_text(draw, (x, y + THUMB_SIZE[1] + 5), f"t{frame_index + 1:02d}", small_font, fill=MUTED)

    if len(path) >= 2:
        for row_index in (0, 1):
            y = HEADER_HEIGHT + row_index * row_height
            points = []
            for candidate in path:
                x = PADDING + ROW_LABEL_WIDTH + candidate.frame_index * cell_width
                cx, cy = scale_point(candidate, source_size)
                points.append((x + cx, y + cy))
            draw.line(points, fill=SELECTED, width=2)
    return sheet


def export_one(rgb_path: Path, output_root: Path, args) -> bool:
    """Export one debug sheet."""
    sequence = np.load(rgb_path).astype(np.float32)
    diff = temporal_diff(sequence)
    color_score = baseball_color_score(sequence)
    candidates = component_candidates(
        sequence=sequence,
        diff=diff,
        color_score=color_score,
        min_area=args.min_area,
        max_area=args.max_area,
        min_motion=args.min_motion,
        min_color=args.min_color,
        min_y=args.min_y,
        max_y=args.max_y,
        max_candidates_per_frame=args.max_candidates_per_frame,
    )
    path = link_candidates(candidates)
    relative = rgb_path.relative_to(args.variant_root / "rgb")
    split, label, filename = relative.parts
    output_path = output_root / split / label / f"{Path(filename).stem}.jpg"
    if output_path.exists() and not args.overwrite:
        return False

    sheet = render_sheet(rgb_path, args.variant_root, sequence, diff, candidates, path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ball candidate trajectory debug sheets")
    parser.add_argument("--variant-root", type=Path, default=VARIANTS_DIR, help="Variant root containing rgb sequences")
    parser.add_argument("--output-dir", type=Path, default=CONTACT_DIR, help="Output contact sheet root")
    parser.add_argument("--split", choices=SPLITS, default=None, help="Optional split to export")
    parser.add_argument("--limit", type=int, default=50, help="Maximum sheets to export")
    parser.add_argument("--min-area", type=int, default=1, help="Minimum connected-component area")
    parser.add_argument("--max-area", type=int, default=36, help="Maximum connected-component area")
    parser.add_argument("--min-motion", type=float, default=0.18, help="Minimum normalized motion score")
    parser.add_argument("--min-color", type=float, default=0.12, help="Minimum baseball-color score")
    parser.add_argument("--min-y", type=float, default=0.05, help="Minimum normalized y candidate position")
    parser.add_argument("--max-y", type=float, default=0.88, help="Maximum normalized y candidate position")
    parser.add_argument("--max-candidates-per-frame", type=int, default=10, help="Keep top candidates per frame")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing debug sheets")
    args = parser.parse_args()

    ensure_modeling_dirs()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rgb_paths = list_rgb_sequences(args.variant_root, args.split, args.limit)
    print(f"Found {len(rgb_paths)} RGB sequence(s) to inspect")
    print(f"Writing ball-track debug sheets to: {args.output_dir}")

    exported = 0
    skipped = 0
    for rgb_path in tqdm(rgb_paths, desc="Exporting ball-track debug sheets"):
        did_export = export_one(rgb_path, args.output_dir, args)
        if did_export:
            exported += 1
        else:
            skipped += 1

    print(f"Exported {exported} ball-track debug sheet(s)")
    if skipped:
        print(f"Skipped existing sheet(s): {skipped}")


if __name__ == "__main__":
    main()
