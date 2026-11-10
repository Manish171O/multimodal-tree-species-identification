"""Trust-weighted adaptive fusion of modality embeddings with missing-sensor masking."""
from __future__ import annotations

import torch
import torch.nn as nn


class AdaptiveFusion(nn.Module):
    """Fuse M embeddings (B, M, D) into (B, D).

    mode="trust": w_m = mask_m · (t_m + ε)^γ / Σ_k mask_k · (t_k + ε)^γ  (trust from QualityEstimator)
    mode="mean" : w_m = mask_m / Σ_k mask_k                              (ablation: no quality estimator)

    Missing modalities always get weight exactly 0, so their (garbage or zero) embeddings cannot
    leak into the result. If a sample has no modality at all the fused embedding is 0.
    ``detach_trust`` stops the classification loss from steering the trust scores, so trust is
    learned only from the evidential loss (it stays an honest per-sensor reliability signal).
    """

    def __init__(self, embed_dim: int, mode: str = "trust", gamma: float = 1.0, eps: float = 1e-3,
                 detach_trust: bool = True) -> None:
        super().__init__()
        if mode not in ("trust", "mean"):
            raise ValueError(f"unknown fusion mode '{mode}'")
        self.mode, self.gamma, self.eps, self.detach_trust = mode, gamma, eps, detach_trust
        self.post = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim), nn.GELU())

    def weights(self, modality_mask: torch.Tensor, trust: torch.Tensor | None) -> torch.Tensor:
        """modality_mask: (B, M), trust: (B, M) or None -> normalized weights (B, M)."""
        if self.mode == "mean" or trust is None:
            raw = modality_mask                                                  # (B, M)
        else:
            t = trust.detach() if self.detach_trust else trust                   # (B, M)
            raw = modality_mask * (t + self.eps).pow(self.gamma)                 # (B, M)
        return raw / raw.sum(1, keepdim=True).clamp(min=1e-8)                    # (B, M) rows sum to 1 (or 0)

    def forward(self, embs: torch.Tensor, modality_mask: torch.Tensor, trust: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """embs: (B, M, D), modality_mask: (B, M), trust: (B, M) -> fused (B, D), weights (B, M)."""
        w = self.weights(modality_mask, trust)                                   # (B, M)
        fused = (w.unsqueeze(-1) * embs).sum(1)                                  # (B, D)
        any_present = (modality_mask.sum(1, keepdim=True) > 0).to(fused.dtype)   # (B, 1)
        return self.post(fused) * any_present, w                                 # (B, D), (B, M)
