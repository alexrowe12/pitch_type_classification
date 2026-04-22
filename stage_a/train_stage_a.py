#!/usr/bin/env python3
"""
Train the Stage A single-frame shot classifier.

Usage:
    python -m stage_a.train_stage_a
    python -m stage_a.train_stage_a --epochs 5 --batch-size 32
"""

import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from stage_a.paths import STAGE_A_METRICS_JSON, STAGE_A_MODEL_PT, TRAIN_LABELS_CSV, ensure_stage_a_dirs


LABEL_TO_INDEX = {"non_pitch_camera": 0, "pitch_camera": 1}
INDEX_TO_LABEL = {index: label for label, index in LABEL_TO_INDEX.items()}
IMAGE_SIZE = 224


def load_train_rows(path: Path) -> list[dict]:
    """Load merged train-label rows."""
    if not path.exists():
        raise FileNotFoundError(f"Missing train labels at {path}. Run stage_a.prepare_train_labels first.")
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


class StageADataset(Dataset):
    """Simple image dataset for Stage A classification."""

    def __init__(self, rows: list[dict], transform):
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = Image.open(row["frame_path"]).convert("RGB")
        image_tensor = self.transform(image)
        label = LABEL_TO_INDEX[row["label"]]
        return image_tensor, label


def build_transforms(train: bool):
    """Return the transform pipeline for train/eval."""
    if train:
        return transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def split_rows(rows: list[dict], val_ratio: float) -> tuple[list[dict], list[dict]]:
    """Split rows into train/val, using stratification when viable."""
    labels = [row["label"] for row in rows]
    unique_labels = sorted(set(labels))

    if len(rows) < 10 or len(unique_labels) < 2:
        cutoff = max(1, int(len(rows) * (1.0 - val_ratio)))
        return rows[:cutoff], rows[cutoff:]

    label_counts = {label: labels.count(label) for label in unique_labels}
    can_stratify = all(count >= 2 for count in label_counts.values())

    train_rows, val_rows = train_test_split(
        rows,
        test_size=val_ratio,
        random_state=42,
        stratify=labels if can_stratify else None,
        shuffle=True,
    )
    return train_rows, val_rows


def build_model() -> nn.Module:
    """Build a lightweight transfer-learning model."""
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(LABEL_TO_INDEX))
    return model


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / max(1, total_examples)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    """Evaluate on a validation loader."""
    model.eval()
    y_true = []
    y_pred = []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        preds = torch.argmax(logits, dim=1).cpu().tolist()
        y_pred.extend(preds)
        y_true.extend(labels.tolist())

    if not y_true:
        return {"accuracy": 0.0, "confusion_matrix": [[0, 0], [0, 0]]}

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def save_artifacts(model, metrics: dict) -> None:
    """Persist model weights and training metrics."""
    STAGE_A_MODEL_PT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), STAGE_A_MODEL_PT)

    with open(STAGE_A_METRICS_JSON, "w") as handle:
        json.dump(metrics, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Stage A shot-classification model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Optimizer learning rate")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    args = parser.parse_args()

    ensure_stage_a_dirs()
    rows = load_train_rows(TRAIN_LABELS_CSV)
    if len(rows) < 2:
        raise ValueError("Need at least 2 labeled rows to train Stage A.")
    labels_present = {row["label"] for row in rows}
    if labels_present != set(LABEL_TO_INDEX):
        raise ValueError(
            "Stage A training requires both labels to be present in train_labels.csv. "
            f"Found only: {sorted(labels_present)}"
        )

    train_rows, val_rows = split_rows(rows, args.val_ratio)
    if not val_rows:
        raise ValueError("Validation split is empty. Add more labels or increase dataset size.")

    train_dataset = StageADataset(train_rows, transform=build_transforms(train=True))
    val_dataset = StageADataset(val_rows, transform=build_transforms(train=False))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)

    label_counts = {label: sum(1 for row in train_rows if row["label"] == label) for label in LABEL_TO_INDEX}
    total_train = sum(label_counts.values())
    class_weights = torch.tensor(
        [total_train / max(1, label_counts[INDEX_TO_LABEL[i]]) for i in range(len(LABEL_TO_INDEX))],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    history = []
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(epoch_metrics)
        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_accuracy={val_metrics['accuracy']:.4f}"
        )

    final_val_metrics = evaluate(model, val_loader, device)
    metrics = {
        "num_rows": len(rows),
        "num_train_rows": len(train_rows),
        "num_val_rows": len(val_rows),
        "device": str(device),
        "history": history,
        "final_val_accuracy": final_val_metrics["accuracy"],
        "final_val_confusion_matrix": final_val_metrics["confusion_matrix"],
        "label_to_index": LABEL_TO_INDEX,
    }
    save_artifacts(model, metrics)

    print(f"Saved model to: {STAGE_A_MODEL_PT}")
    print(f"Saved metrics to: {STAGE_A_METRICS_JSON}")


if __name__ == "__main__":
    main()
