"""Shared PyTorch utilities for Stage A scripts."""

import torch


def select_device(preferred: str = "auto") -> torch.device:
    """Select the best available PyTorch device."""
    preferred = preferred.lower()

    if preferred != "auto":
        if preferred == "mps" and not torch.backends.mps.is_available():
            raise ValueError("Requested MPS device, but torch.backends.mps is not available.")
        if preferred == "cuda" and not torch.cuda.is_available():
            raise ValueError("Requested CUDA device, but torch.cuda.is_available() is false.")
        return torch.device(preferred)

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def should_pin_memory(device: torch.device) -> bool:
    """Pin DataLoader memory only for CUDA, where it helps host-to-device transfer."""
    return device.type == "cuda"
