"""Batched inference that collects everything the evaluation needs."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.model import MultimodalTreeNet


def corrupt_batch(batch: dict[str, torch.Tensor], kind: str, severity: float, g: torch.Generator
                  ) -> dict[str, torch.Tensor]:
    """Test-time sensor corruption on normalized inputs (the sensor stays 'present' in the mask).

    kinds: 'rgb_noise' / 'hsi_noise' — additive Gaussian noise with std = ``severity`` (z-units);
           'lidar_noise' — xyz jitter with std = ``severity`` and 80 % of points removed.
    These corruptions are NOT used in training; they test whether trust-weighted fusion
    down-weights a sensor that silently became unreliable.
    """
    out = dict(batch)
    if kind in ("rgb_noise", "hsi_noise"):
        k = kind.split("_")[0]
        x = batch[k]
        n = torch.randn(x.shape, generator=g).to(x.device) * severity                   # same shape as x
        out[k] = x + n * batch["modality_mask"][:, 0 if k == "rgb" else 1].view(-1, 1, 1, 1)
    elif kind == "lidar_noise":
        pts, m = batch["lidar"], batch["lidar_mask"]                                    # (B, P, 4), (B, P)
        jit = torch.randn(pts[..., :3].shape, generator=g).to(pts.device) * severity    # (B, P, 3)
        keep = (torch.rand(m.shape, generator=g).to(m.device) < 0.2) & m                # (B, P)
        keep[:, 0] |= m[:, 0]                                                           # keep >= 1 point
        out["lidar"] = torch.cat([pts[..., :3] + jit, pts[..., 3:]], -1) * keep.unsqueeze(-1)
        out["lidar_mask"] = keep
    else:
        raise ValueError(f"unknown corruption '{kind}'")
    return out


@torch.no_grad()
def predict(model: MultimodalTreeNet, loader: DataLoader, device: torch.device,
            keep: Sequence[int] | None = None, amp: bool = False, corrupt: str | None = None,
            severity: float = 1.0, seed: int = 0) -> dict[str, np.ndarray]:
    """Run ``model`` over ``loader``. ``keep`` = (rgb, hsi, lidar) 0/1 drops sensors for every sample;
    ``corrupt`` applies a seeded test-time corruption (see :func:`corrupt_batch`).

    Returns arrays with N rows: label, pred, prob (N, C), trust/weights/mask/uncertainty (N, 3),
    structure + struct_true (N, 2, metres), site, index.
    """
    model.eval()
    keys = ("label", "pred", "prob", "trust", "uncertainty", "weights", "mask", "structure", "struct_true",
            "site", "index")
    acc: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    g = torch.Generator().manual_seed(seed)
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        if corrupt:
            batch = corrupt_batch(batch, corrupt, severity, g)
        b = batch["label"].shape[0]
        k = None if keep is None else torch.tensor(list(keep), dtype=torch.float32, device=device).expand(b, 3)  # (B, 3)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            out = model(batch, keep=k)
        prob = torch.softmax(out["logits"].float(), 1)                                                  # (B, C)
        acc["label"].append(batch["label"].cpu().numpy())
        acc["pred"].append(prob.argmax(1).cpu().numpy())
        acc["prob"].append(prob.cpu().numpy())
        for name in ("trust", "uncertainty", "weights"):
            acc[name].append(out[name].float().cpu().numpy())
        acc["mask"].append(out["modality_mask"].cpu().numpy())
        acc["structure"].append(model.structure_metres(out).float().cpu().numpy())                      # (B, 2)
        acc["struct_true"].append(torch.stack([batch["height"], batch["crown_diameter"]], 1).cpu().numpy())
        acc["site"].append(batch["site"].cpu().numpy())
        acc["index"].append(batch["index"].cpu().numpy())
    return {k: np.concatenate(v) for k, v in acc.items()}
