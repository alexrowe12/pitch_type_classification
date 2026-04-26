#!/usr/bin/env python3
"""
Create a reproducible validation split from Stage B training sequences.

Usage:
    python -m stage_b.create_val_split
    python -m stage_b.create_val_split --val-ratio 0.2 --seed 42
"""

import argparse
import random
import shutil
from pathlib import Path

from stage_b.paths import SEQUENCES_DIR, ensure_stage_b_dirs


LABELS = ["fastball", "offspeed"]


def list_sequence_files(split: str, label: str) -> list[Path]:
    """Return sorted sequence files for one split/label directory."""
    directory = SEQUENCES_DIR / split / label
    if not directory.exists():
        return []
    return sorted(directory.glob("*.npy"))


def restore_val_into_train() -> None:
    """Move any existing val files back into train so the split is reproducible."""
    for label in LABELS:
        val_dir = SEQUENCES_DIR / "val" / label
        train_dir = SEQUENCES_DIR / "train" / label
        train_dir.mkdir(parents=True, exist_ok=True)
        if not val_dir.exists():
            continue
        for path in sorted(val_dir.glob("*.npy")):
            shutil.move(str(path), str(train_dir / path.name))


def choose_val_files(train_files: list[Path], val_ratio: float, seed: int) -> list[Path]:
    """Choose a stratified validation subset for one label."""
    if not train_files:
        return []
    if len(train_files) == 1:
        return []

    val_count = max(1, int(round(len(train_files) * val_ratio)))
    val_count = min(val_count, len(train_files) - 1)
    rng = random.Random(seed)
    shuffled = train_files[:]
    rng.shuffle(shuffled)
    return sorted(shuffled[:val_count], key=lambda path: path.name)


def move_files_to_val(files: list[Path], label: str) -> None:
    """Move selected training files into the validation directory."""
    val_dir = SEQUENCES_DIR / "val" / label
    val_dir.mkdir(parents=True, exist_ok=True)
    for path in files:
        shutil.move(str(path), str(val_dir / path.name))


def summarize_counts() -> dict[str, dict[str, int]]:
    """Return split/label counts after splitting."""
    summary = {}
    for split in ("train", "val", "test"):
        summary[split] = {}
        for label in LABELS:
            summary[split][label] = len(list_sequence_files(split, label))
    return summary


def print_summary(summary: dict[str, dict[str, int]]) -> None:
    """Print split summary."""
    print("Sequence split summary:")
    for split in ("train", "val", "test"):
        total = sum(summary[split].values())
        print(f"  split={split}: {total}")
        for label in LABELS:
            print(f"    {label}: {summary[split][label]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a validation split from Stage B training sequences")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Fraction of train sequences to move into val")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the split")
    args = parser.parse_args()

    ensure_stage_b_dirs()
    restore_val_into_train()

    for label in LABELS:
        train_files = list_sequence_files("train", label)
        val_files = choose_val_files(train_files, val_ratio=args.val_ratio, seed=args.seed)
        move_files_to_val(val_files, label)

    print_summary(summarize_counts())


if __name__ == "__main__":
    main()
