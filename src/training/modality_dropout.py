"""Modality dropout: randomly remove sensors during training so the model degrades gracefully."""
from __future__ import annotations

import torch


def sample_modality_keep(modality_mask: torch.Tensor, p_apply: float, p_drop: tuple[float, float, float],
                         generator: torch.Generator | None = None) -> torch.Tensor:
    """Return a keep mask (B, M) in {0, 1}.

    With probability ``p_apply`` per sample, each modality m is dropped independently with
    probability ``p_drop[m]``. A sample never loses its *last* present sensor: if dropout would
    remove all of them, one present sensor (chosen uniformly) is kept.

    modality_mask: (B, M) — sensors actually present (already-missing ones stay missing).
    """
    b, m = modality_mask.shape
    dev = modality_mask.device
    apply = torch.rand(b, 1, generator=generator).to(dev) < p_apply                     # (B, 1)
    pd = torch.tensor(p_drop, dtype=torch.float32, device=dev).view(1, m)               # (1, M)
    drop = (torch.rand(b, m, generator=generator).to(dev) < pd) & apply                 # (B, M)
    keep = (~drop).float()                                                              # (B, M)
    eff = keep * modality_mask                                                          # (B, M)
    dead = (eff.sum(1) == 0) & (modality_mask.sum(1) > 0)                               # (B,)
    if dead.any():
        scores = torch.rand(b, m, generator=generator).to(dev) * modality_mask          # (B, M) only present
        pick = scores.argmax(1)                                                         # (B,)
        keep[dead] = 0.0
        keep[dead, pick[dead]] = 1.0
    return keep                                                                         # (B, M)
