#!/usr/bin/env python3
"""
Evaluate a saved binary pitch-classifier run.

Usage:
    python -m modeling.evaluate_binary --run-id <run_id> --split val
    python -m modeling.evaluate_binary --run-id <run_id> --split test
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from torch import nn

from modeling.dataset import PitchSequenceDataset, list_variant_rows
from modeling.models import build_model
from modeling.paths import RUNS_DIR, VARIANTS_DIR, ensure_modeling_dirs
from modeling.train_binary import build_loader, evaluate, save_json, save_predictions_csv
from stage_a.torch_utils import select_device, should_pin_memory


def load_checkpoint(path: Path, device: torch.device) -> dict:
    """Load a saved training checkpoint."""
    if not path.exists():
        raise FileNotFoundError(f"Missing model checkpoint at {path}")
    return torch.load(path, map_location=device, weights_only=False)


def load_run_metrics(run_dir: Path) -> dict:
    """Load metrics JSON if present."""
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return {}
    with open(metrics_path) as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved binary pitch classifier")
    parser.add_argument("--run-id", required=True, help="Run ID under data/modeling/runs")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val", help="Split to evaluate")
    parser.add_argument("--variant-root", type=Path, default=VARIANTS_DIR, help="Variant dataset root")
    parser.add_argument("--batch-size", type=int, default=8, help="Evaluation batch size")
    parser.add_argument(
        "--device",
        choices=["auto", "mps", "cuda", "cpu"],
        default="auto",
        help="Device to use. Defaults to auto, preferring Apple MPS, then CUDA, then CPU.",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count")
    args = parser.parse_args()

    ensure_modeling_dirs()
    device = select_device(args.device)
    run_dir = RUNS_DIR / args.run_id
    checkpoint = load_checkpoint(run_dir / "best_model.pt", device=device)
    variant = checkpoint["variant"]
    model_name = checkpoint["model"]
    input_channels = int(checkpoint["input_channels"])

    rows = list_variant_rows(variant, args.split, variant_root=args.variant_root)
    if not rows:
        raise ValueError(f"No rows found for variant={variant}, split={args.split}")

    dataset = PitchSequenceDataset(rows)
    loader = build_loader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=should_pin_memory(device),
    )

    model = build_model(
        model_name,
        input_channels=input_channels,
        dropout=float(checkpoint.get("dropout", 0.35)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = nn.CrossEntropyLoss()
    metrics = evaluate(model, loader, criterion, device)

    run_metrics = load_run_metrics(run_dir)
    payload = {
        "run_id": args.run_id,
        "split": args.split,
        "variant": variant,
        "model": model_name,
        "input_channels": input_channels,
        "device": str(device),
        "num_samples": len(dataset),
        "training_summary": {
            "train_samples": run_metrics.get("train_samples"),
            "val_samples": run_metrics.get("val_samples"),
            "selection_device": run_metrics.get("selection_device"),
            "best_val_metrics": run_metrics.get("best_val_metrics"),
        },
        "metrics": {key: value for key, value in metrics.items() if key != "predictions"},
    }

    metrics_path = run_dir / f"evaluation_{args.split}.json"
    predictions_path = run_dir / f"predictions_{args.split}.csv"
    save_json(metrics_path, payload)
    save_predictions_csv(predictions_path, metrics["predictions"])

    print(f"Using device: {device}")
    print(f"Run: {args.run_id}")
    print(f"Split: {args.split}, samples={len(dataset)}")
    print(f"accuracy={metrics['accuracy']:.4f}")
    print(f"precision={metrics['precision']:.4f}")
    print(f"recall={metrics['recall']:.4f}")
    print(f"f1={metrics['f1']:.4f}")
    print(f"confusion_matrix={metrics['confusion_matrix']}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved predictions to: {predictions_path}")


if __name__ == "__main__":
    main()
