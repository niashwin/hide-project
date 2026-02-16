"""
Cross-modal projection for shared embedding space.

Trains lightweight projection layers (~10K params) to align independently
pre-trained text and image encoders into a shared space.
"""

import torch
import torch.nn as nn
from typing import Optional


class ModalityProjection(nn.Module):
    """Linear projection from a modality-specific space to the shared HIDE space."""

    def __init__(self, input_dim: int, output_dim: int = 512):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.projection(x))


class CrossModalSpace(nn.Module):
    """Shared cross-modal embedding space with modality-specific projections."""

    def __init__(self, text_dim: int = 384, image_dim: int = 512,
                 shared_dim: int = 512):
        super().__init__()
        self.text_proj = ModalityProjection(text_dim, shared_dim)
        self.image_proj = ModalityProjection(image_dim, shared_dim)
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def encode_text(self, text_features: torch.Tensor) -> torch.Tensor:
        proj = self.text_proj(text_features)
        return proj / proj.norm(dim=-1, keepdim=True)

    def encode_image(self, image_features: torch.Tensor) -> torch.Tensor:
        proj = self.image_proj(image_features)
        return proj / proj.norm(dim=-1, keepdim=True)

    def symmetric_infonce_loss(self, text_features: torch.Tensor,
                                image_features: torch.Tensor) -> torch.Tensor:
        """Symmetric InfoNCE loss for cross-modal alignment."""
        text_proj = self.encode_text(text_features)
        image_proj = self.encode_image(image_features)

        logits = (text_proj @ image_proj.T) / self.temperature.exp()
        labels = torch.arange(len(logits), device=logits.device)

        loss_t2i = nn.functional.cross_entropy(logits, labels)
        loss_i2t = nn.functional.cross_entropy(logits.T, labels)
        return (loss_t2i + loss_i2t) / 2
