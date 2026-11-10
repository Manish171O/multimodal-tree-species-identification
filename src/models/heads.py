"""Prediction heads: species classifier, auxiliary structure regressor, contrastive projections."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeciesHead(nn.Module):
    """Fused embedding (B, D) -> species logits (B, C)."""

    def __init__(self, embed_dim: int, num_classes: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(embed_dim, num_classes))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, D) -> logits (B, C)."""
        return self.net(z)                        # (B, C)


class StructureHead(nn.Module):
    """Fused embedding (B, D) -> standardized [height, crown_diameter] (B, 2).

    Targets are z-scored with train-split statistics in the loss; ``denormalize`` maps back to metres.
    """

    def __init__(self, embed_dim: int, hidden: int = 64, n_targets: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(embed_dim, hidden), nn.GELU(), nn.Linear(hidden, n_targets))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, D) -> (B, 2)."""
        return self.net(z)                        # (B, 2)

    @staticmethod
    def denormalize(pred: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """pred: (B, 2), mean/std: (2,) -> metres (B, 2)."""
        return pred * std + mean                  # (B, 2)


class ProjectionHead(nn.Module):
    """Embedding (B, D) -> L2-normalized projection (B, P) for the contrastive loss."""

    def __init__(self, embed_dim: int, proj_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, proj_dim))

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        """e: (B, D) -> (B, P) unit-norm rows."""
        return F.normalize(self.net(e), dim=-1)   # (B, P)
