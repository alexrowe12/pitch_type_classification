"""Model definitions for binary pitch classification."""

import torch
from torch import nn
from torchvision import models


def norm_2d(kind: str, channels: int) -> nn.Module:
    """Return a 2D normalization layer."""
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    if kind == "group":
        groups = min(8, channels)
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"Unknown normalization kind: {kind}")


def norm_3d(kind: str, channels: int) -> nn.Module:
    """Return a 3D normalization layer."""
    if kind == "batch":
        return nn.BatchNorm3d(channels)
    if kind == "group":
        groups = min(8, channels)
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"Unknown normalization kind: {kind}")


class Small3DCNN(nn.Module):
    """Compact 3D CNN baseline for short pitch sequences."""

    def __init__(self, input_channels: int, num_classes: int = 2, dropout: float = 0.35, norm: str = "batch"):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(input_channels, 16, kernel_size=3, padding=1, bias=False),
            norm_3d(norm, 16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(16, 32, kernel_size=3, padding=1, bias=False),
            norm_3d(norm, 32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=3, padding=1, bias=False),
            norm_3d(norm, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Conv3d(64, 128, kernel_size=3, padding=1, bias=False),
            norm_3d(norm, 128),
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

    def __init__(self, input_channels: int, num_classes: int = 2, dropout: float = 0.35, norm: str = "batch"):
        super().__init__()
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1, bias=False),
            norm_2d(norm, 16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            norm_2d(norm, 32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            norm_2d(norm, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            norm_2d(norm, 128),
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


class ResNet18FramePool(nn.Module):
    """Transfer-learning baseline: frozen ResNet-18 frame encoder plus temporal pooling."""

    def __init__(self, input_channels: int, num_classes: int = 2, dropout: float = 0.35):
        super().__init__()
        if input_channels != 3:
            raise ValueError("resnet18_frame_pool requires the rgb variant with 3 input channels")

        weights = models.ResNet18_Weights.DEFAULT
        backbone = models.resnet18(weights=weights)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        for parameter in backbone.parameters():
            parameter.requires_grad = False

        self.backbone = backbone
        self.register_buffer("image_mean", torch.tensor(weights.transforms().mean).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor(weights.transforms().std).view(1, 3, 1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return class logits for RGB inputs shaped (B, 3, T, H, W)."""
        batch_size, channels, frames, height, width = inputs.shape
        frame_inputs = inputs.permute(0, 2, 1, 3, 4).reshape(batch_size * frames, channels, height, width)
        frame_inputs = (frame_inputs - self.image_mean) / self.image_std
        frame_features = self.backbone(frame_inputs)
        frame_features = frame_features.reshape(batch_size, frames, -1)
        mean_features = frame_features.mean(dim=1)
        max_features = frame_features.max(dim=1).values
        sequence_features = torch.cat([mean_features, max_features], dim=1)
        return self.classifier(sequence_features)


def build_model(model_name: str, input_channels: int, dropout: float = 0.35) -> nn.Module:
    """Build a model by name."""
    if model_name == "small_3d_cnn":
        return Small3DCNN(input_channels=input_channels, dropout=dropout)
    if model_name == "small_3d_cnn_gn":
        return Small3DCNN(input_channels=input_channels, dropout=dropout, norm="group")
    if model_name == "frame_cnn_pool":
        return FrameCNNPool(input_channels=input_channels, dropout=dropout)
    if model_name == "frame_cnn_pool_gn":
        return FrameCNNPool(input_channels=input_channels, dropout=dropout, norm="group")
    if model_name == "resnet18_frame_pool":
        return ResNet18FramePool(input_channels=input_channels, dropout=dropout)
    raise ValueError(f"Unknown model: {model_name}")
