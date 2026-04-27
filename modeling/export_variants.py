#!/usr/bin/env python3
"""
Export derived sequence variants for binary pitch classification.

Usage:
    python -m modeling.export_variants
    python -m modeling.export_variants --variants rgb diff rgb_diff
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from modeling.audit_dataset import LABELS, SPLITS
from modeling.paths import VARIANTS_DIR, ensure_modeling_dirs
from stage_b.paths import SEQUENCES_DIR


DEFAULT_VARIANTS = ("rgb", "diff", "rgb_diff")
DEFAULT_WORKERS = min(8, os.cpu_count() or 1)


def list_source_sequences(source_dir: Path) -> list[Path]:
    """Return all source sequence files in split/label order."""
    paths = []
    for split in SPLITS:
        for label in LABELS:
            paths.extend(sorted((source_dir / split / label).glob("*.npy")))
    return paths


def relative_sequence_path(path: Path, source_dir: Path) -> Path:
    """Return split/label/file relative path for a source sequence."""
    return path.relative_to(source_dir)


def grayscale_from_rgb(sequence: np.ndarray) -> np.ndarray:
    """Convert RGB sequence to one-channel grayscale."""
    red = sequence[..., 0]
    green = sequence[..., 1]
    blue = sequence[..., 2]
    gray = (0.299 * red) + (0.587 * green) + (0.114 * blue)
    return gray[..., np.newaxis].astype(np.float32)


def temporal_diff(sequence: np.ndarray) -> np.ndarray:
    """Return one-channel absolute frame-difference motion sequence."""
    gray = grayscale_from_rgb(sequence)
    diffs = np.zeros_like(gray, dtype=np.float32)
    diffs[1:] = np.abs(gray[1:] - gray[:-1])

    max_value = float(diffs.max())
    if max_value > 0:
        diffs = diffs / max_value
    return diffs.astype(np.float32)


def build_variant(sequence: np.ndarray, variant: str) -> np.ndarray:
    """Build one model input variant from an RGB sequence."""
    if variant == "rgb":
        return sequence.astype(np.float32)
    if variant == "diff":
        return temporal_diff(sequence)
    if variant == "rgb_diff":
        return np.concatenate([sequence.astype(np.float32), temporal_diff(sequence)], axis=-1)
    raise ValueError(f"Unknown variant: {variant}")


def save_variant(
    source_path: Path,
    source_dir: Path,
    output_root: Path,
    variant: str,
    overwrite: bool,
) -> tuple[str, str | None]:
    """Save one variant file and return (variant, error_reason)."""
    relative_path = relative_sequence_path(source_path, source_dir)
    output_path = output_root / variant / relative_path
    if output_path.exists() and not overwrite:
        return variant, None

    try:
        sequence = np.load(source_path)
        variant_sequence = build_variant(sequence, variant)
    except Exception:  # noqa: BLE001 - report per-file failures after parallel export.
        return variant, "variant_failed"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, variant_sequence)
    return variant, None


def export_source_sequence(
    source_path: Path,
    source_dir: Path,
    output_root: Path,
    variants: tuple[str, ...],
    overwrite: bool,
) -> list[tuple[str, str | None]]:
    """Export all requested variants for one sequence."""
    return [
        save_variant(
            source_path=source_path,
            source_dir=source_dir,
            output_root=output_root,
            variant=variant,
            overwrite=overwrite,
        )
        for variant in variants
    ]


def summarize_variant_counts(output_root: Path, variants: tuple[str, ...]) -> None:
    """Print exported counts grouped by variant/split/label."""
    print("Variant export summary:")
    for variant in variants:
        print(f"  variant={variant}")
        for split in SPLITS:
            split_total = 0
            label_counts = {}
            for label in LABELS:
                count = len(list((output_root / variant / split / label).glob("*.npy")))
                label_counts[label] = count
                split_total += count
            print(f"    {split}: {split_total}")
            for label, count in label_counts.items():
                print(f"      {label}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export binary model sequence variants")
    parser.add_argument("--source-dir", type=Path, default=SEQUENCES_DIR, help="Source Stage B sequence root")
    parser.add_argument("--output-root", type=Path, default=VARIANTS_DIR, help="Variant output root")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=DEFAULT_VARIANTS,
        default=list(DEFAULT_VARIANTS),
        help="Variants to export",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum source sequences to export")
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of sequence export workers (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing variant files")
    args = parser.parse_args()

    ensure_modeling_dirs()
    source_paths = list_source_sequences(args.source_dir)
    if args.limit is not None:
        source_paths = source_paths[: args.limit]

    variants = tuple(args.variants)
    print(f"Found {len(source_paths)} source sequence(s)")
    print(f"Exporting variants: {', '.join(variants)}")
    print(f"Using workers={args.workers}")

    skipped: dict[str, int] = {}
    exported_counts = {variant: 0 for variant in variants}

    def export_one(path: Path) -> list[tuple[str, str | None]]:
        return export_source_sequence(
            source_path=path,
            source_dir=args.source_dir,
            output_root=args.output_root,
            variants=variants,
            overwrite=args.overwrite,
        )

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for results in tqdm(
            executor.map(export_one, source_paths),
            total=len(source_paths),
            desc="Exporting modeling variants",
        ):
            for variant, error_reason in results:
                if error_reason is not None:
                    skipped[error_reason] = skipped.get(error_reason, 0) + 1
                    continue
                exported_counts[variant] += 1

    for variant, count in sorted(exported_counts.items()):
        print(f"Processed {count} {variant} variant file(s)")
    if skipped:
        print("Skipped variant files:")
        for reason, count in sorted(skipped.items()):
            print(f"  {reason}: {count}")

    summarize_variant_counts(args.output_root, variants)


if __name__ == "__main__":
    main()
