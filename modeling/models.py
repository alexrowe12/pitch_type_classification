"""Model definitions for binary pitch classification."""

import torch
from torch import nn


class Small3DCNN(nn.Module):
    """Compact 3D CNN baseline for short pitch sequences."""

    def __init__(self, input_channels: int, num_classes: int = 2, dropout: float = 0.35):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(input_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Conv3d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return class logits for inputs shaped (B, C, T, H, W)."""
        return self.classifier(self.features(inputs))


def build_model(model_name: str, input_channels: int, dropout: float = 0.35) -> nn.Module:
    """Build a model by name."""
    if model_name == "small_3d_cnn":
        return Small3DCNN(input_channels=input_channels, dropout=dropout)
    raise ValueError(f"Unknown model: {model_name}")
