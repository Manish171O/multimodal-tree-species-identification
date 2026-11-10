"""Focal loss (Lin et al., 2017) with optional class-balanced α (Cui et al., 2019)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def class_balanced_weights(counts: np.ndarray, beta: float = 0.999) -> torch.Tensor:
    """Effective-number weights w_c ∝ (1 − β) / (1 − β^{n_c}), normalized to mean 1 over seen classes.

    counts: (C,) training counts per class (0 allowed -> weight 0). Returns (C,) float32.
    """
    counts = np.asarray(counts, dtype=np.float64)
    eff = 1.0 - np.power(beta, counts)                                      # (C,)
    w = np.where(counts > 0, (1.0 - beta) / np.maximum(eff, 1e-12), 0.0)    # (C,)
    seen = counts > 0
    w[seen] = w[seen] / w[seen].mean()
    return torch.tensor(w, dtype=torch.float32)


class FocalLoss(nn.Module):
    """FL = −α_y (1 − p_y)^γ log p_y, averaged over the batch. γ = 0, α = None reduces to cross-entropy."""

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer("alpha", alpha if alpha is not None else torch.empty(0))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """logits: (B, C), target: (B,) -> scalar."""
        logp = F.log_softmax(logits.float(), dim=-1)                          # (B, C)
        logp_y = logp.gather(1, target.unsqueeze(1)).squeeze(1)               # (B,)
        p_y = logp_y.exp()                                                    # (B,)
        if self.label_smoothing > 0:
            ce = (1 - self.label_smoothing) * -logp_y - self.label_smoothing * logp.mean(-1)  # (B,)
        else:
            ce = -logp_y                                                      # (B,)
        loss = (1 - p_y).clamp(min=0).pow(self.gamma) * ce                    # (B,)
        if self.alpha.numel():
            a = self.alpha[target]                                            # (B,)
            return (a * loss).sum() / a.sum().clamp(min=1e-8)
        return loss.mean()
