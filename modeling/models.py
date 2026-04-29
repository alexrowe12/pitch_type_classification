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


class FrameCNNPool(nn.Module):
    """2D frame encoder with temporal average/max pooling."""

    def __init__(self, input_channels: int, num_classes: int = 2, dropout: float = 0.35):
        super().__init__()
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128 * 2, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(96, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return class logits for inputs shaped (B, C, T, H, W)."""
        batch_size, channels, frames, height, width = inputs.shape
        frame_inputs = inputs.permute(0, 2, 1, 3, 4).reshape(batch_size * frames, channels, height, width)
        frame_features = self.frame_encoder(frame_inputs).flatten(1)
        frame_features = frame_features.reshape(batch_size, frames, -1)
        mean_features = frame_features.mean(dim=1)
        max_features = frame_features.max(dim=1).values
        sequence_features = torch.cat([mean_features, max_features], dim=1)
        return self.classifier(sequence_features)


def build_model(model_name: str, input_channels: int, dropout: float = 0.35) -> nn.Module:
    """Build a model by name."""
    if model_name == "small_3d_cnn":
        return Small3DCNN(input_channels=input_channels, dropout=dropout)
    if model_name == "frame_cnn_pool":
        return FrameCNNPool(input_channels=input_channels, dropout=dropout)
    raise ValueError(f"Unknown model: {model_name}")
