"""Cross-modal contrastive (InfoNCE) loss: pull together embeddings of the same tree across sensors."""
from __future__ import annotations

from itertools import combinations

import torch
import torch.nn.functional as F


def pairwise_infonce(za: torch.Tensor, zb: torch.Tensor, temperature: float = 0.1,
                     labels: torch.Tensor | None = None) -> torch.Tensor:
    """Symmetric InfoNCE between two views. za, zb: (N, P) unit-norm; labels: (N,) optional.

    Row i of ``za`` and row i of ``zb`` are the positive pair (same tree). If ``labels`` is given,
    other trees of the *same species* are removed from the negatives (avoid false negatives).
    """
    logits = za @ zb.t() / temperature                                         # (N, N)
    if labels is not None:
        same = labels.unsqueeze(0) == labels.unsqueeze(1)                      # (N, N)
        same.fill_diagonal_(False)
        logits = logits.masked_fill(same, float("-inf"))                       # (N, N)
    tgt = torch.arange(za.shape[0], device=za.device)                          # (N,)
    return 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.t(), tgt))  # ()


def cross_modal_contrastive_loss(proj: dict[str, torch.Tensor], modality_mask: torch.Tensor,
                                 modalities: tuple[str, ...], temperature: float = 0.1,
                                 labels: torch.Tensor | None = None) -> torch.Tensor:
    """Average InfoNCE over all modality pairs, using only samples where both modalities exist.

    proj: {m: (B, P)} unit-norm projections; modality_mask: (B, M). Pairs with < 2 valid samples
    are skipped; returns 0 (with a graph) if no pair is usable.
    """
    losses: list[torch.Tensor] = []
    for i, j in combinations(range(len(modalities)), 2):
        valid = (modality_mask[:, i] > 0) & (modality_mask[:, j] > 0)          # (B,)
        if valid.sum() < 2:
            continue
        lab = labels[valid] if labels is not None else None                    # (N,)
        losses.append(pairwise_infonce(proj[modalities[i]][valid], proj[modalities[j]][valid], temperature, lab))
    if not losses:
        return sum(p.sum() for p in proj.values()) * 0.0                       # () keeps autograd happy
    return torch.stack(losses).mean()                                          # ()
