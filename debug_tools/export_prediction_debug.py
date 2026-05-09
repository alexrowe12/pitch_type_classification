#!/usr/bin/env python3
"""
Export prediction debug sheets for a trained binary model run.

Usage:
    python -m modeling.export_prediction_debug --run-id baseline_diff_30 --split val
"""

import argparse
import csv
import json
from pathlib import Path

from tqdm import tqdm

from modeling.export_variant_debug import render_contact_sheet
from modeling.paths import DEBUG_DIR, RUNS_DIR, VARIANTS_DIR, ensure_modeling_dirs


def load_metrics(run_dir: Path) -> dict:
    """Load run metrics."""
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file at {metrics_path}")
    with open(metrics_path) as handle:
        return json.load(handle)


def load_predictions(path: Path) -> list[dict]:
    """Load prediction rows."""
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions CSV at {path}")
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        row["offspeed_probability"] = float(row["offspeed_probability"])
        row["correct"] = row["true_label"] == row["pred_label"]
        row["confidence"] = (
            row["offspeed_probability"]
            if row["pred_label"] == "offspeed"
            else 1.0 - row["offspeed_probability"]
        )
        row["uncertainty"] = abs(row["offspeed_probability"] - 0.5)
    return rows


def sort_predictions(rows: list[dict], sort_mode: str) -> list[dict]:
    """Sort prediction rows for review."""
    if sort_mode == "wrong-first":
        return sorted(rows, key=lambda row: (row["correct"], -row["confidence"], row["clip_id"]))
    if sort_mode == "uncertain-first":
        return sorted(rows, key=lambda row: (row["uncertainty"], row["clip_id"]))
    if sort_mode == "confident-first":
        return sorted(rows, key=lambda row: (-row["confidence"], row["clip_id"]))
    return sorted(rows, key=lambda row: row["clip_id"])


def rgb_variant_path(variant_root: Path, split: str, true_label: str, clip_id: str) -> Path:
    """Return the RGB variant path for a prediction row."""
    return variant_root / "rgb" / split / true_label / f"{clip_id}.npy"


def output_category(row: dict) -> str:
    """Return the review category for a prediction row."""
    if not row["correct"]:
        return "wrong"
    if row["uncertainty"] <= 0.10:
        return "uncertain_correct"
    return "correct"


def output_filename(index: int, row: dict) -> str:
    """Build a sortable output filename."""
    probability = int(round(row["offspeed_probability"] * 1000))
    return (
        f"{index:03d}_{row['clip_id']}_"
        f"true-{row['true_label']}_pred-{row['pred_label']}_p{probability:03d}.jpg"
    )


def prediction_subtitle(row: dict) -> str:
    """Return prediction metadata for sheet header."""
    status = "correct" if row["correct"] else "wrong"
    return (
        f"{status} | true={row['true_label']} pred={row['pred_label']} "
        f"offspeed_prob={row['offspeed_probability']:.3f} confidence={row['confidence']:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export prediction debug contact sheets")
    parser.add_argument("--run-id", required=True, help="Run ID under data/modeling/runs")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val", help="Prediction split")
    parser.add_argument("--variant-root", type=Path, default=VARIANTS_DIR, help="Variant dataset root")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEBUG_DIR / "prediction_contacts",
        help="Prediction contact sheet output root",
    )
    parser.add_argument(
        "--sort",
        choices=["wrong-first", "uncertain-first", "confident-first", "clip-id"],
        default="wrong-first",
        help="Review order",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum sheets to export")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing sheets")
    args = parser.parse_args()

    ensure_modeling_dirs()
    run_dir = RUNS_DIR / args.run_id
    metrics = load_metrics(run_dir)
    predictions_path = run_dir / f"predictions_{args.split}.csv"
    rows = sort_predictions(load_predictions(predictions_path), args.sort)
    if args.limit is not None:
        rows = rows[: args.limit]

    output_root = args.output_dir / args.run_id / args.split
    output_root.mkdir(parents=True, exist_ok=True)
    variant = metrics.get("variant", "unknown")
    print(f"Run: {args.run_id}")
    print(f"Variant: {variant}")
    print(f"Split: {args.split}")
    print(f"Found {len(rows)} prediction row(s) to render")

    exported = 0
    skipped = 0
    missing = 0
    for index, row in enumerate(tqdm(rows, desc="Exporting prediction debug sheets"), start=1):
        rgb_path = rgb_variant_path(args.variant_root, args.split, row["true_label"], row["clip_id"])
        if not rgb_path.exists():
            missing += 1
            continue

        category = output_category(row)
        output_path = output_root / category / output_filename(index, row)
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue

        sheet = render_contact_sheet(rgb_path, args.variant_root, subtitle=prediction_subtitle(row))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(output_path, quality=92)
        exported += 1

    print(f"Exported {exported} prediction debug sheet(s) to: {output_root}")
    if skipped:
        print(f"Skipped existing sheet(s): {skipped}")
    if missing:
        print(f"Missing RGB variant source(s): {missing}")


if __name__ == "__main__":
    main()
