"""Inference for one or many tree crowns: species (top-k), height, crown diameter, per-sensor trust."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from src.data.crown_dataset import CrownPreprocessor
from src.evaluation.evaluate import load_model
from src.utils.common import get_device

MODS = ("rgb", "hsi", "lidar")


@dataclass
class Prediction:
    """Everything predicted for one crown (and its truth, when known)."""

    top_species: list[tuple[str, str, float]]    # (common name, scientific name, probability), best first
    height_m: float
    crown_diameter_m: float
    trust: dict[str, float]                       # evidential trust per sensor (0 = missing)
    weight: dict[str, float]                      # fusion weight per sensor
    sensors_used: list[str]
    true_species: str | None = None
    true_height_m: float | None = None
    true_crown_m: float | None = None
    source: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def species(self) -> str:
        return self.top_species[0][0]

    @property
    def confidence(self) -> float:
        return self.top_species[0][2]

    @property
    def correct(self) -> bool | None:
        return None if self.true_species is None else self.species == self.true_species


class TreePredictor:
    """Loads a checkpoint trained on the demo_crowns layout and predicts from raw sensor arrays."""

    def __init__(self, checkpoint: str | Path, device: str = "auto", top_k: int = 5) -> None:
        self.device = get_device(device)
        self.model, self.cfg, ck = load_model(checkpoint, self.device)
        self.stats = ck["norm_stats"]
        if self.stats.get("format") != "demo_crowns":
            raise ValueError("predict.py needs a model trained on the demo_crowns layout "
                             "(train with configs/real_species.yaml)")
        self.names = self.stats["class_names"]
        self.scientific = self.stats["class_scientific"]
        self.prep = CrownPreprocessor(self.stats, self.cfg["data"])
        self.top_k = min(top_k, len(self.names))
        self.epoch = int(ck["epoch"])

    @torch.no_grad()
    def predict(self, crowns: Sequence[tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]],
                drop: Sequence[str] = (), seed: int = 0) -> list[Prediction]:
        """crowns: list of (rgb (3,H,W) uint8 | None, hsi (B,h,w) reflectance | None, chm (H,W) m | None)."""
        rng = np.random.default_rng(seed)
        items = [self.prep(r, h, c, rng=rng, augment=False) for r, h, c in crowns]
        batch = {k: torch.stack([it[k] for it in items]).to(self.device) for k in items[0]}
        keep = torch.tensor([[0.0 if m in drop else 1.0 for m in MODS]], device=self.device).expand(len(items), 3)
        out = self.model(batch, keep=keep)
        prob = torch.softmax(out["logits"].float(), 1).cpu().numpy()                     # (N, C)
        struct = self.model.structure_metres(out).float().cpu().numpy()                    # (N, 2)
        trust, w = out["trust"].float().cpu().numpy(), out["weights"].float().cpu().numpy()  # (N, 3)
        mask = out["modality_mask"].cpu().numpy()
        res = []
        for i in range(len(items)):
            order = np.argsort(-prob[i])[: self.top_k]
            res.append(Prediction(
                top_species=[(self.names[c], self.scientific[c], float(prob[i, c])) for c in order],
                height_m=float(max(struct[i, 0], 0.0)), crown_diameter_m=float(max(struct[i, 1], 0.0)),
                trust={m: float(trust[i, j]) for j, m in enumerate(MODS)},
                weight={m: float(w[i, j]) for j, m in enumerate(MODS)},
                sensors_used=[m for j, m in enumerate(MODS) if mask[i, j] > 0]))
        return res


# --------------------------------------------------------------------------- user-file loading
def _resize(a: np.ndarray, hw: tuple[int, int], nearest: bool = False) -> np.ndarray:
    """Resize the last two axes of ``a`` to ``hw`` (bilinear, or nearest for height rasters)."""
    h, w = a.shape[-2:]
    if (h, w) == tuple(hw):
        return a
    ys = (np.arange(hw[0]) + 0.5) * h / hw[0] - 0.5
    xs = (np.arange(hw[1]) + 0.5) * w / hw[1] - 0.5
    if nearest:
        yi = np.clip(np.round(ys).astype(int), 0, h - 1)
        xi = np.clip(np.round(xs).astype(int), 0, w - 1)
        return a[..., yi, :][..., xi]                                  # rows, then columns
    y0 = np.clip(np.floor(ys).astype(int), 0, h - 1); y1 = np.clip(y0 + 1, 0, h - 1); fy = np.clip(ys - y0, 0, 1)
    x0 = np.clip(np.floor(xs).astype(int), 0, w - 1); x1 = np.clip(x0 + 1, 0, w - 1); fx = np.clip(xs - x0, 0, 1)
    top = a[..., y0, :][..., x0] * (1 - fx) + a[..., y0, :][..., x1] * fx
    bot = a[..., y1, :][..., x0] * (1 - fx) + a[..., y1, :][..., x1] * fx
    return top * (1 - fy)[:, None] + bot * fy[:, None]


def load_user_rgb(path: str | Path, size: Sequence[int]) -> np.ndarray:
    """PNG/JPG/.npy crown image -> (3, H, W) uint8 at the model's resolution."""
    p = Path(path)
    if p.suffix.lower() == ".npy":
        a = np.load(p)
        a = a if a.shape[0] == 3 else np.moveaxis(a, -1, 0)
    else:
        from PIL import Image                                     # installed with matplotlib
        a = np.moveaxis(np.asarray(Image.open(p).convert("RGB")), -1, 0)
    a = _resize(a.astype(np.float32), tuple(size[-2:]))
    return np.clip(np.round(a), 0, 255).astype(np.uint8)


def load_user_hsi(path: str | Path, shape: Sequence[int]) -> np.ndarray:
    """.npy cube (B, h, w) or single spectrum (B,) of reflectance -> (B, h, w) at the model's size.

    Values above 2 are taken to be scaled by 10000 (NEON convention) and divided back to reflectance.
    """
    a = np.load(path).astype(np.float32)
    if a.ndim == 1:
        a = np.broadcast_to(a[:, None, None], (a.shape[0],) + tuple(shape[-2:])).copy()
    if a.shape[0] != shape[0]:
        raise ValueError(f"hyperspectral input must have {shape[0]} bands, got {a.shape[0]}")
    if np.nanmax(a) > 2:
        a = a / 10000.0
    return _resize(a, tuple(shape[-2:]))


def load_user_lidar(path: str | Path, shape: Sequence[int]) -> np.ndarray:
    """.npy canopy-height raster in metres (H, W) or (1, H, W) -> (H, W) at the model's size."""
    a = np.load(path).astype(np.float32)
    a = a.reshape(a.shape[-2:])
    return _resize(a, tuple(shape[-2:]), nearest=True)
