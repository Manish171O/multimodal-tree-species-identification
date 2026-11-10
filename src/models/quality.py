"""Evidential per-modality, per-sample quality (trust) estimator.

Each modality gets a small evidential head (Sensoy et al., 2018) that maps its embedding to
non-negative class evidence e ∈ R^C. The Dirichlet parameters are α = e + 1, total strength
S = Σ α, and the subjective-logic uncertainty is u = C / S ∈ (0, 1]. Trust is t = 1 − u:
a sensor that gathers little evidence for *any* class (noisy, hazy, sparse, or uninformative
for this tree) gets low trust. A missing sensor gets trust exactly 0 via ``modality_mask``.

This is not a softmax weight: trust is the Dirichlet evidence mass, trained by the evidential
loss (src/losses/evidential.py) on each modality's own prediction. Weights are only
normalized afterwards, in the fusion module.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidentialHead(nn.Module):
    """Embedding (B, D) -> Dirichlet concentration α (B, C), α ≥ 1."""

    def __init__(self, embed_dim: int, num_classes: int, hidden: int = 128, activation: str = "exp",
                 max_log_evidence: float = 10.0) -> None:
        super().__init__()
        if activation not in ("exp", "softplus"):
            raise ValueError(f"unknown evidence activation '{activation}'")
        self.activation, self.max_log_evidence = activation, max_log_evidence
        self.net = nn.Sequential(nn.Linear(embed_dim, hidden), nn.GELU(), nn.Linear(hidden, num_classes))

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """emb: (B, D) -> alpha: (B, C).

        ``exp`` evidence (clamped) lets evidence grow multiplicatively, which matters with many
        classes: with softplus, total evidence stays tiny next to C = 167 and trust collapses
        towards 0 for every sensor (observed in Phase 4 runs).
        """
        z = self.net(emb).float()                                     # (B, C)
        if self.activation == "exp":
            evidence = torch.exp(z.clamp(max=self.max_log_evidence))  # (B, C)  non-negative evidence
        else:
            evidence = F.softplus(z)                                  # (B, C)
        return evidence + 1.0                                         # (B, C)  Dirichlet α


class QualityEstimator(nn.Module):
    """Computes per-modality Dirichlet α, uncertainty and trust for M modalities."""

    def __init__(self, modalities: tuple[str, ...], embed_dim: int, num_classes: int, hidden: int = 128,
                 activation: str = "exp") -> None:
        super().__init__()
        self.modalities = modalities
        self.num_classes = num_classes
        self.heads = nn.ModuleDict({m: EvidentialHead(embed_dim, num_classes, hidden, activation) for m in modalities})

    def forward(self, embs: dict[str, torch.Tensor], modality_mask: torch.Tensor
                ) -> dict[str, torch.Tensor]:
        """embs: {m: (B, D)}, modality_mask: (B, M) -> dict with
        alpha (B, M, C), uncertainty (B, M), trust (B, M) — trust is 0 where the mask is 0."""
        alpha = torch.stack([self.heads[m](embs[m]) for m in self.modalities], dim=1)  # (B, M, C)
        strength = alpha.sum(-1)                                                        # (B, M)
        unc = self.num_classes / strength                                               # (B, M) in (0, 1]
        trust = (1.0 - unc) * modality_mask                                             # (B, M)
        unc = torch.where(modality_mask > 0, unc, torch.ones_like(unc))                 # missing -> u = 1
        return {"alpha": alpha, "uncertainty": unc, "trust": trust}
