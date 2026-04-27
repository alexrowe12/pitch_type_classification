#!/usr/bin/env python3
"""
Train a binary pitch classifier from modeling sequence variants.

Usage:
    python -m modeling.train_binary --variant rgb --epochs 30
    python -m modeling.train_binary --variant rgb --overfit-samples 16 --epochs 50
"""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, Subset

from modeling.dataset import INDEX_TO_LABEL, LABEL_TO_INDEX, PitchSequenceDataset, list_variant_rows
from modeling.models import build_model
from modeling.paths import RUNS_DIR, VARIANTS_DIR, ensure_modeling_dirs
from stage_a.torch_utils import select_device, should_pin_memory


DEFAULT_BATCH_SIZE = 8
DEFAULT_EPOCHS = 30


def make_run_id(variant: str, model_name: str) -> str:
    """Return a timestamped run identifier."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{variant}_{model_name}"


def set_seed(seed: int) -> None:
    """Set random seeds for repeatable small-data experiments."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_input_channels(variant: str, variant_root: Path) -> int:
    """Read one sequence file to infer channel count."""
    rows = list_variant_rows(variant, "train", variant_root=variant_root)
    if not rows:
        raise ValueError(f"No training rows found for variant={variant} under {variant_root}")
    sequence = np.load(rows[0]["path"], mmap_mode="r")
    if sequence.ndim != 4:
        raise ValueError(f"Expected sequence shape (T,H,W,C), got {sequence.shape}")
    return int(sequence.shape[-1])


def build_loader(
    dataset: PitchSequenceDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
):
    """Build a DataLoader for sequence datasets."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def load_datasets(variant: str, variant_root: Path) -> dict[str, PitchSequenceDataset]:
    """Load train/val/test datasets."""
    return {
        split: PitchSequenceDataset(list_variant_rows(variant, split, variant_root=variant_root))
        for split in ("train", "val", "test")
    }


def restrict_for_overfit(dataset: PitchSequenceDataset, count: int) -> PitchSequenceDataset | Subset:
    """Return a small class-balanced subset for overfit sanity checks."""
    if count <= 0:
        return dataset

    indices_by_label = {label_index: [] for label_index in INDEX_TO_LABEL}
    for index, row in enumerate(dataset.rows):
        indices_by_label[row["label_index"]].append(index)

    selected = []
    while len(selected) < min(count, len(dataset)):
        added_this_round = False
        for label_index in sorted(indices_by_label):
            if indices_by_label[label_index] and len(selected) < count:
                selected.append(indices_by_label[label_index].pop(0))
                added_this_round = True
        if not added_this_round:
            break

    indices = sorted(selected)
    return Subset(dataset, indices)


def class_weights(dataset) -> torch.Tensor:
    """Compute simple inverse-frequency class weights."""
    counts = {index: 0 for index in INDEX_TO_LABEL}
    for item_index in range(len(dataset)):
        row = dataset.dataset.rows[dataset.indices[item_index]] if isinstance(dataset, Subset) else dataset.rows[item_index]
        counts[row["label_index"]] += 1
    total = sum(counts.values())
    weights = [total / max(1, counts[index]) for index in sorted(counts)]
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_examples = 0
    for inputs, labels, _clip_ids in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
    return total_loss / max(1, total_examples)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> dict:
    """Evaluate a model and return metrics plus per-clip predictions."""
    model.eval()
    y_true = []
    y_pred = []
    y_prob = []
    clip_ids = []
    total_loss = 0.0
    total_examples = 0

    for inputs, labels, batch_clip_ids in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = model(inputs)
        loss = criterion(logits, labels)
        probabilities = torch.softmax(logits, dim=1)
        preds = torch.argmax(probabilities, dim=1)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
        y_prob.extend(probabilities[:, LABEL_TO_INDEX["offspeed"]].cpu().tolist())
        clip_ids.extend(batch_clip_ids)

    if not y_true:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "confusion_matrix": [[0, 0], [0, 0]],
            "predictions": [],
        }

    predictions = [
        {
            "clip_id": clip_id,
            "true_label": INDEX_TO_LABEL[true_index],
            "pred_label": INDEX_TO_LABEL[pred_index],
            "offspeed_probability": float(probability),
        }
        for clip_id, true_index, pred_index, probability in zip(clip_ids, y_true, y_pred, y_prob)
    ]
    return {
        "loss": total_loss / max(1, total_examples),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "predictions": predictions,
    }


def save_json(path: Path, payload: dict) -> None:
    """Save a JSON payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def save_predictions_csv(path: Path, predictions: list[dict]) -> None:
    """Save predictions in CSV form for inspection."""
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["clip_id", "true_label", "pred_label", "offspeed_probability"],
        )
        writer.writeheader()
        writer.writerows(predictions)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a binary pitch classifier")
    parser.add_argument("--variant", choices=["rgb", "diff", "rgb_diff"], default="rgb", help="Input variant")
    parser.add_argument("--variant-root", type=Path, default=VARIANTS_DIR, help="Variant dataset root")
    parser.add_argument("--model", choices=["small_3d_cnn"], default="small_3d_cnn", help="Model architecture")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Optimizer learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Optimizer weight decay")
    parser.add_argument("--dropout", type=float, default=0.35, help="Classifier dropout rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--overfit-samples", type=int, default=0, help="Use N train samples for overfit sanity check")
    parser.add_argument("--run-id", type=str, default=None, help="Optional run identifier")
    parser.add_argument(
        "--device",
        choices=["auto", "mps", "cuda", "cpu"],
        default="auto",
        help="Device to use. Defaults to auto, preferring Apple MPS, then CUDA, then CPU.",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count")
    args = parser.parse_args()

    ensure_modeling_dirs()
    set_seed(args.seed)
    datasets = load_datasets(args.variant, args.variant_root)
    if len(datasets["train"]) == 0 or len(datasets["val"]) == 0:
        raise ValueError("Training requires non-empty train and val datasets. Run variant export and val split first.")

    train_dataset = restrict_for_overfit(datasets["train"], args.overfit_samples)
    val_dataset = train_dataset if args.overfit_samples else datasets["val"]

    device = select_device(args.device)
    pin_memory = should_pin_memory(device)
    input_channels = infer_input_channels(args.variant, args.variant_root)
    model = build_model(args.model, input_channels=input_channels, dropout=args.dropout).to(device)
    weights = class_weights(train_dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    train_loader = build_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = build_loader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    run_id = args.run_id or make_run_id(args.variant, args.model)
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Run: {run_id}")
    print(f"Variant: {args.variant}, input_channels={input_channels}")
    print(f"Train samples: {len(train_dataset)}, val samples: {len(val_dataset)}")

    best_f1 = -1.0
    history = []
    best_metrics = {}
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_metrics = evaluate(model, train_loader, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_f1": train_metrics["f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
        }
        history.append(epoch_metrics)

        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_metrics['accuracy']:.3f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.3f} "
            f"val_f1={val_metrics['f1']:.3f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_metrics = val_metrics
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "variant": args.variant,
                    "model": args.model,
                    "input_channels": input_channels,
                    "label_to_index": LABEL_TO_INDEX,
                    "args": vars(args),
                    "dropout": args.dropout,
                },
                run_dir / "best_model.pt",
            )

    metrics = {
        "run_id": run_id,
        "variant": args.variant,
        "model": args.model,
        "input_channels": input_channels,
        "dropout": args.dropout,
        "seed": args.seed,
        "device": str(device),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "overfit_samples": args.overfit_samples,
        "history": history,
        "best_val_metrics": {key: value for key, value in best_metrics.items() if key != "predictions"},
        "label_to_index": LABEL_TO_INDEX,
    }
    save_json(run_dir / "metrics.json", metrics)
    save_predictions_csv(run_dir / "predictions_val.csv", best_metrics.get("predictions", []))

    print(f"Saved best model to: {run_dir / 'best_model.pt'}")
    print(f"Saved metrics to: {run_dir / 'metrics.json'}")
    print(f"Saved validation predictions to: {run_dir / 'predictions_val.csv'}")


if __name__ == "__main__":
    main()
