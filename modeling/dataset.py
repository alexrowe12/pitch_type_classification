"""PyTorch datasets for binary pitch classification."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from modeling.audit_dataset import LABELS, SPLITS
from modeling.paths import VARIANTS_DIR


LABEL_TO_INDEX = {"fastball": 0, "offspeed": 1}
INDEX_TO_LABEL = {index: label for label, index in LABEL_TO_INDEX.items()}


def list_variant_rows(variant: str, split: str, variant_root: Path = VARIANTS_DIR) -> list[dict]:
    """Return rows for one variant/split."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")

    rows = []
    for label in LABELS:
        directory = variant_root / variant / split / label
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.npy")):
            rows.append(
                {
                    "clip_id": path.stem,
                    "label": label,
                    "label_index": LABEL_TO_INDEX[label],
                    "path": path,
                    "split": split,
                    "variant": variant,
                }
            )
    return rows


class PitchSequenceDataset(Dataset):
    """Dataset for fixed-length pitch sequence arrays."""

    def __init__(self, rows: list[dict], cache_data: bool = False):
        self.rows = rows
        self.cached_items = [self.load_item(row) for row in rows] if cache_data else None

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def load_item(row: dict):
        sequence = np.load(row["path"]).astype(np.float32)
        tensor = torch.from_numpy(sequence).permute(3, 0, 1, 2).contiguous()
        label = torch.tensor(row["label_index"], dtype=torch.long)
        return tensor, label, row["clip_id"]

    def __getitem__(self, index: int):
        if self.cached_items is not None:
            return self.cached_items[index]
        return self.load_item(self.rows[index])


def load_variant_datasets(
    variant: str,
    variant_root: Path = VARIANTS_DIR,
    cache_data: bool = False,
) -> dict[str, PitchSequenceDataset]:
    """Load train/val/test datasets for a variant."""
    return {
        split: PitchSequenceDataset(list_variant_rows(variant, split, variant_root=variant_root), cache_data=cache_data)
        for split in SPLITS
    }
