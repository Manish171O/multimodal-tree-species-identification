"""Combined objective: focal classification + contrastive + auxiliary structure + evidential."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from src.losses.contrastive import cross_modal_contrastive_loss
from src.losses.evidential import evidential_loss
from src.losses.focal import FocalLoss


def kl_anneal(epoch: int, anneal_epochs: int, kl_max: float) -> float:
    """Linear warm-up of the evidential KL weight: 0 at epoch 0 -> kl_max at ``anneal_epochs``."""
    if anneal_epochs <= 0:
        return kl_max
    return kl_max * min(1.0, epoch / anneal_epochs)


class TotalLoss(torch.nn.Module):
    """L = w_cls·Focal + w_con·InfoNCE + w_aux·SmoothL1(height, crown) + w_evi·Evidential.

    Weights and options come from the ``loss:`` config section. ``forward`` returns a dict of
    detached components plus the differentiable ``total``.
    """

    def __init__(self, lcfg: dict[str, Any], modalities: tuple[str, ...], alpha: torch.Tensor | None = None) -> None:
        super().__init__()
        self.c = lcfg
        self.modalities = modalities
        self.focal = FocalLoss(float(lcfg["focal_gamma"]), alpha, float(lcfg.get("label_smoothing", 0.0)))

    def forward(self, out: dict[str, Any], batch: dict[str, torch.Tensor], target_mean: torch.Tensor,
                target_std: torch.Tensor, epoch: int = 0) -> dict[str, torch.Tensor]:
        c = self.c
        y = batch["label"]                                                              # (B,)
        cls = self.focal(out["logits"], y)                                              # ()
        con = cross_modal_contrastive_loss(
            out["proj"], out["modality_mask"], self.modalities, float(c["contrastive_temperature"]),
            y if c.get("contrastive_exclude_same_class", True) else None)               # ()
        tgt = torch.stack([batch["height"], batch["crown_diameter"]], dim=1)            # (B, 2)
        tgt = (tgt - target_mean) / target_std                                          # (B, 2) z-scored
        aux = F.smooth_l1_loss(out["structure"].float(), tgt)                           # ()
        klw = kl_anneal(epoch, int(c.get("evidential_anneal_epochs", 10)), float(c["evidential_kl_max"]))
        evi = evidential_loss(out["alpha"].float(), y, out["modality_mask"], klw)       # ()
        total = (c["w_cls"] * cls + c["w_contrastive"] * con + c["w_aux"] * aux + c["w_evidential"] * evi)
        return {"total": total, "cls": cls.detach(), "contrastive": con.detach(), "aux": aux.detach(),
                "evidential": evi.detach(), "kl_weight": torch.tensor(klw)}
