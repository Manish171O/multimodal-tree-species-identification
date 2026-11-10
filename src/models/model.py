"""Full model: encoders -> evidential quality estimator -> adaptive fusion -> species + structure heads."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.data.dataset import MODALITIES
from src.models.encoders import build_encoders
from src.models.fusion import AdaptiveFusion
from src.models.heads import ProjectionHead, SpeciesHead, StructureHead
from src.models.quality import QualityEstimator


def apply_modality_mask(batch: dict[str, torch.Tensor], keep: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return a shallow copy of ``batch`` with modalities zeroed where ``keep`` (B, 3) is 0.

    The effective mask is batch['modality_mask'] * keep, so a sensor that is already missing
    stays missing. Used for modality dropout (training) and missing-modality evaluation.
    """
    out = dict(batch)
    m = batch["modality_mask"] * keep.to(batch["modality_mask"].dtype)          # (B, 3)
    out["modality_mask"] = m
    out["rgb"] = batch["rgb"] * m[:, 0].view(-1, 1, 1, 1)                       # (B, 3, H, W)
    out["hsi"] = batch["hsi"] * m[:, 1].view(-1, 1, 1, 1)                       # (B, K, h, w)
    out["lidar"] = batch["lidar"] * m[:, 2].view(-1, 1, 1)                      # (B, P, 4)
    out["lidar_mask"] = batch["lidar_mask"] & (m[:, 2:3] > 0)                   # (B, P)
    return out


class MultimodalTreeNet(nn.Module):
    """Adaptive quality-aware multimodal tree species classifier.

    Forward output dict (B = batch, M = 3 modalities, C = classes, D = embed_dim, P = proj_dim):
        logits      (B, C)     species logits from the fused embedding
        structure   (B, 2)     standardized [height, crown_diameter]
        fused       (B, D)
        embeddings  (B, M, D)  per-modality encoder outputs
        proj        {m: (B, P)} unit-norm projections for the contrastive loss
        alpha       (B, M, C)  per-modality Dirichlet concentrations
        uncertainty (B, M), trust (B, M), weights (B, M), modality_mask (B, M)
    """

    def __init__(self, cfg: dict[str, Any], num_classes: int, hsi_bands: int, lidar_features: int = 4,
                 target_mean: tuple[float, float] = (0.0, 0.0), target_std: tuple[float, float] = (1.0, 1.0)
                 ) -> None:
        super().__init__()
        m = cfg["model"]
        d = int(m["embed_dim"])
        self.modalities: tuple[str, ...] = MODALITIES
        self.num_classes = num_classes
        self.use_quality = bool(m.get("use_quality_estimator", True))
        self.encoders = build_encoders(m, hsi_bands, lidar_features)
        self.quality = QualityEstimator(self.modalities, d, num_classes, int(m.get("quality_hidden", 128)),
                                        str(m.get("evidence_activation", "exp")))
        self.fusion = AdaptiveFusion(d, "trust" if self.use_quality else "mean", float(m.get("trust_gamma", 1.0)),
                                     detach_trust=bool(m.get("detach_trust", True)))
        self.species_head = SpeciesHead(d, num_classes, float(m.get("head_dropout", 0.2)))
        self.structure_head = StructureHead(d)
        self.proj = nn.ModuleDict({k: ProjectionHead(d, int(m.get("proj_dim", 128))) for k in self.modalities})
        self.register_buffer("target_mean", torch.tensor(target_mean, dtype=torch.float32))  # (2,)
        self.register_buffer("target_std", torch.tensor(target_std, dtype=torch.float32))    # (2,)

    def encode(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run the three encoders -> {m: (B, D)}."""
        return {"rgb": self.encoders["rgb"](batch["rgb"]),                            # (B, D)
                "hsi": self.encoders["hsi"](batch["hsi"]),                            # (B, D)
                "lidar": self.encoders["lidar"](batch["lidar"], batch["lidar_mask"])}  # (B, D)

    def forward(self, batch: dict[str, torch.Tensor], keep: torch.Tensor | None = None) -> dict[str, Any]:
        """batch: dataset dict; keep: optional (B, 3) mask of sensors to keep (dropout / eval)."""
        if keep is not None:
            batch = apply_modality_mask(batch, keep)
        mask = batch["modality_mask"]                                                  # (B, M)
        embs = self.encode(batch)                                                      # {m: (B, D)}
        q = self.quality(embs, mask)                                                   # alpha (B,M,C), trust (B,M)
        stacked = torch.stack([embs[k] for k in self.modalities], dim=1)               # (B, M, D)
        fused, w = self.fusion(stacked, mask, q["trust"] if self.use_quality else None)  # (B, D), (B, M)
        return {
            "logits": self.species_head(fused),                                        # (B, C)
            "structure": self.structure_head(fused),                                   # (B, 2)
            "fused": fused, "embeddings": stacked, "weights": w, "modality_mask": mask,
            "proj": {k: self.proj[k](embs[k]) for k in self.modalities},               # {m: (B, P)}
            "alpha": q["alpha"], "uncertainty": q["uncertainty"], "trust": q["trust"],
        }

    def structure_metres(self, out: dict[str, Any]) -> torch.Tensor:
        """Structure prediction in metres, (B, 2)."""
        return StructureHead.denormalize(out["structure"], self.target_mean, self.target_std)  # (B, 2)


def build_model(cfg: dict[str, Any], train_ds: Any) -> MultimodalTreeNet:
    """Build the model with class count, band count and target stats taken from the training data."""
    sample = train_ds[0]
    st = train_ds.stats
    return MultimodalTreeNet(cfg, num_classes=train_ds.num_classes, hsi_bands=int(sample["hsi"].shape[0]),
                             lidar_features=int(sample["lidar"].shape[1]),
                             target_mean=(st["height_mean"], st["crown_mean"]),
                             target_std=(st["height_std"], st["crown_std"]))
