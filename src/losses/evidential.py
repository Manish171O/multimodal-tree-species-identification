"""Evidential (Dirichlet) loss that trains the per-modality quality estimator (Sensoy et al., 2018)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def kl_dirichlet_uniform(alpha: torch.Tensor) -> torch.Tensor:
    """KL( Dir(alpha) || Dir(1) ). alpha: (..., C) -> (...)."""
    c = alpha.shape[-1]
    s = alpha.sum(-1)                                                          # (...)
    return (torch.lgamma(s) - torch.lgamma(torch.tensor(float(c), device=alpha.device))
            - torch.lgamma(alpha).sum(-1)
            + ((alpha - 1) * (torch.digamma(alpha) - torch.digamma(s).unsqueeze(-1))).sum(-1))  # (...)


def evidential_loss(alpha: torch.Tensor, target: torch.Tensor, modality_mask: torch.Tensor,
                    kl_weight: float = 0.0) -> torch.Tensor:
    """Expected cross-entropy under the Dirichlet + annealed KL on misleading evidence.

    Args:
        alpha: (B, M, C) Dirichlet concentrations per modality.
        target: (B,) int64 class ids.
        modality_mask: (B, M) 1 = modality present; missing ones contribute nothing.
        kl_weight: λ_t, usually annealed from 0 to its max over the first epochs.

    Returns:
        scalar loss averaged over present (sample, modality) pairs (0 if none).
    """
    y = F.one_hot(target, alpha.shape[-1]).to(alpha.dtype).unsqueeze(1)       # (B, 1, C)
    s = alpha.sum(-1, keepdim=True)                                            # (B, M, 1)
    nll = (y * (torch.digamma(s) - torch.digamma(alpha))).sum(-1)              # (B, M)
    alpha_tilde = y + (1 - y) * alpha                                          # (B, M, C) remove true-class evidence
    kl = kl_dirichlet_uniform(alpha_tilde)                                     # (B, M)
    per = (nll + kl_weight * kl) * modality_mask                               # (B, M)
    return per.sum() / modality_mask.sum().clamp(min=1.0)                      # ()
