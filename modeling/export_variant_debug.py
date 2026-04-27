#!/usr/bin/env python3
"""
Export visual debug sheets for binary modeling sequence variants.

Usage:
    python -m modeling.export_variant_debug --limit 50
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from modeling.audit_dataset import LABELS, SPLITS
from modeling.paths import DEBUG_DIR, VARIANTS_DIR, ensure_modeling_dirs


CONTACT_DIR = DEBUG_DIR / "variant_contacts"
THUMB_SIZE = (112, 112)
PADDING = 10
HEADER_HEIGHT = 76
ROW_LABEL_WIDTH = 96
ROW_GAP = 18
LABEL_HEIGHT = 24
BACKGROUND = (18, 22, 26)
CARD_FILL = (30, 37, 43)
TEXT = (238, 241, 245)
MUTED = (161, 170, 181)
RGB_BORDER = (91, 156, 220)
DIFF_BORDER = (245, 184, 82)
OVERLAY_BORDER = (118, 191, 137)


def list_rgb_sequences(variant_root: Path, split: str | None, limit: int | None) -> list[Path]:
    """Return RGB variant paths in split/label order."""
    splits = [split] if split else list(SPLITS)
    paths = []
    for split_name in splits:
        for label in LABELS:
            paths.extend(sorted((variant_root / "rgb" / split_name / label).glob("*.npy")))
    if limit is not None:
        paths = paths[:limit]
    return paths


def variant_path_for(rgb_path: Path, variant_root: Path, variant: str) -> Path:
    """Return the corresponding path for another variant."""
    relative_path = rgb_path.relative_to(variant_root / "rgb")
    return variant_root / variant / relative_path


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


def rgb_frame_to_image(frame: np.ndarray) -> Image.Image:
    """Convert a normalized RGB frame into a PIL image."""
    array = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB").resize(THUMB_SIZE)


def diff_frame_to_image(frame: np.ndarray) -> Image.Image:
    """Convert a normalized one-channel diff frame into a PIL image."""
    array = np.clip(frame[..., 0] * 255.0, 0, 255).astype(np.uint8)
    image = Image.fromarray(array, mode="L").convert("RGB")
    return image.resize(THUMB_SIZE)


def overlay_frame_to_image(rgb_frame: np.ndarray, diff_frame: np.ndarray) -> Image.Image:
    """Overlay diff motion as warm highlights on top of RGB."""
    rgb = np.clip(rgb_frame * 255.0, 0, 255).astype(np.float32)
    diff = np.clip(diff_frame[..., 0], 0.0, 1.0)
    heat = np.zeros_like(rgb)
    heat[..., 0] = 255.0
    heat[..., 1] = 175.0
    heat[..., 2] = 40.0
    alpha = np.clip(diff[..., np.newaxis] * 1.7, 0.0, 0.75)
    overlay = (rgb * (1.0 - alpha)) + (heat * alpha)
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB").resize(THUMB_SIZE)


def draw_frame_row(
    sheet: Image.Image,
    draw: ImageDraw.ImageDraw,
    label: str,
    images: list[Image.Image],
    y: int,
    border: tuple[int, int, int],
    label_font,
    small_font,
) -> int:
    """Draw one row of frame thumbnails and return next y coordinate."""
    draw_text(draw, (PADDING, y + 36), label, label_font)
    x_start = PADDING + ROW_LABEL_WIDTH
    cell_width = THUMB_SIZE[0] + PADDING

    for index, image in enumerate(images):
        x = x_start + index * cell_width
        draw.rectangle(
            [x - 3, y - 3, x + THUMB_SIZE[0] + 3, y + THUMB_SIZE[1] + LABEL_HEIGHT - 3],
            fill=CARD_FILL,
        )
        sheet.paste(image, (x, y))
        draw.rectangle([x, y, x + THUMB_SIZE[0] - 1, y + THUMB_SIZE[1] - 1], outline=border, width=2)
        draw_text(draw, (x, y + THUMB_SIZE[1] + 5), f"t{index + 1:02d}", small_font, fill=MUTED)

    return y + THUMB_SIZE[1] + LABEL_HEIGHT + ROW_GAP


def render_contact_sheet(rgb_path: Path, variant_root: Path, subtitle: str | None = None) -> Image.Image:
    """Render one variant debug sheet."""
    diff_path = variant_path_for(rgb_path, variant_root, "diff")
    if not diff_path.exists():
        raise FileNotFoundError(f"Missing diff variant for {rgb_path}")

    rgb = np.load(rgb_path)
    diff = np.load(diff_path)

    if rgb.shape[:3] != diff.shape[:3]:
        raise ValueError(f"RGB/diff shape mismatch for {rgb_path}: {rgb.shape} vs {diff.shape}")

    relative = rgb_path.relative_to(variant_root / "rgb")
    split, label, filename = relative.parts
    clip_id = Path(filename).stem

    columns = rgb.shape[0]
    width = PADDING * 2 + ROW_LABEL_WIDTH + columns * (THUMB_SIZE[0] + PADDING)
    height = HEADER_HEIGHT + 3 * (THUMB_SIZE[1] + LABEL_HEIGHT + ROW_GAP) + PADDING
    sheet = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(sheet)

    title_font = load_font(22)
    label_font = load_font(15)
    small_font = load_font(12)

    draw_text(draw, (PADDING, PADDING), f"{clip_id} | {split} | {label}", title_font)
    subtitle_text = subtitle or f"rgb={rgb.shape} diff={diff.shape}"
    draw_text(draw, (PADDING, PADDING + 30), subtitle_text, label_font, fill=MUTED)

    rgb_images = [rgb_frame_to_image(frame) for frame in rgb]
    diff_images = [diff_frame_to_image(frame) for frame in diff]
    overlay_images = [overlay_frame_to_image(rgb_frame, diff_frame) for rgb_frame, diff_frame in zip(rgb, diff)]

    y = HEADER_HEIGHT
    y = draw_frame_row(sheet, draw, "RGB", rgb_images, y, RGB_BORDER, label_font, small_font)
    y = draw_frame_row(sheet, draw, "Diff", diff_images, y, DIFF_BORDER, label_font, small_font)
    draw_frame_row(sheet, draw, "Overlay", overlay_images, y, OVERLAY_BORDER, label_font, small_font)
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description="Export debug contact sheets for modeling variants")
    parser.add_argument("--variant-root", type=Path, default=VARIANTS_DIR, help="Modeling variant root")
    parser.add_argument("--output-dir", type=Path, default=CONTACT_DIR, help="Contact sheet output directory")
    parser.add_argument("--split", choices=SPLITS, default=None, help="Optional split to export")
    parser.add_argument("--limit", type=int, default=50, help="Maximum sheets to export")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing contact sheets")
    args = parser.parse_args()

    ensure_modeling_dirs()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rgb_paths = list_rgb_sequences(args.variant_root, split=args.split, limit=args.limit)
    print(f"Found {len(rgb_paths)} RGB sequence(s) to render")

    exported = 0
    skipped = 0
    for rgb_path in tqdm(rgb_paths, desc="Exporting variant debug sheets"):
        relative = rgb_path.relative_to(args.variant_root / "rgb")
        split, label, filename = relative.parts
        output_path = args.output_dir / split / label / f"{Path(filename).stem}.jpg"
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue

        sheet = render_contact_sheet(rgb_path, args.variant_root)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(output_path, quality=92)
        exported += 1

    print(f"Exported {exported} variant debug sheet(s) to: {args.output_dir}")
    if skipped:
        print(f"Skipped existing sheet(s): {skipped}")


if __name__ == "__main__":
    main()
