"""Modality encoders. Each maps its sensor to an embedding of the same size ``embed_dim``.

Design notes
  * GroupNorm / LayerNorm everywhere instead of BatchNorm: missing sensors arrive as all-zero
    tensors, and BatchNorm running statistics would be polluted by them (and break at batch=1).
  * Encoders always return finite embeddings, also for missing inputs; *ignoring* a missing
    sensor is the job of the fusion module (Phase 3), which receives ``modality_mask``.

Shapes use B = batch, D = embed_dim, C = channels, H/W = spatial, K = HSI bands, P = points.
"""
from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(c: int) -> nn.GroupNorm:
    """GroupNorm with up to 8 groups that divide ``c``."""
    g = next(g for g in (8, 4, 2, 1) if c % g == 0)
    return nn.GroupNorm(g, c)


class ResBlock2d(nn.Module):
    """Pre-activation residual block: GN-GELU-Conv3x3 ×2, optional stride-2 downsampling."""

    def __init__(self, c_in: int, c_out: int, stride: int = 1) -> None:
        super().__init__()
        self.n1, self.c1 = _gn(c_in), nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.n2, self.c2 = _gn(c_out), nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.skip = (nn.Identity() if stride == 1 and c_in == c_out
                     else nn.Conv2d(c_in, c_out, 1, stride, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C_in, H, W) -> (B, C_out, H/stride, W/stride)."""
        h = self.c1(F.gelu(self.n1(x)))          # (B, C_out, H', W')
        h = self.c2(F.gelu(self.n2(h)))          # (B, C_out, H', W')
        return h + self.skip(x)                  # (B, C_out, H', W')


# ================================================================================ RGB
class RGBEncoder(nn.Module):
    """Compact ResNet-style CNN for high-resolution RGB crown crops."""

    def __init__(self, embed_dim: int = 128, widths: Sequence[int] = (32, 64, 128, 256),
                 dropout: float = 0.1, in_channels: int = 3) -> None:
        super().__init__()
        self.stem = nn.Conv2d(in_channels, widths[0], 3, 1, 1, bias=False)
        blocks: list[nn.Module] = []
        c = widths[0]
        for i, w in enumerate(widths):
            blocks.append(ResBlock2d(c, w, stride=1 if i == 0 else 2))
            c = w
        self.blocks = nn.Sequential(*blocks)
        self.norm = _gn(c)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(c, embed_dim), nn.LayerNorm(embed_dim))

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """rgb: (B, 3, H, W) -> (B, D)."""
        x = self.stem(rgb)                                   # (B, w0, H, W)
        x = self.blocks(x)                                   # (B, w_last, H/8, W/8)
        x = F.gelu(self.norm(x))                             # (B, w_last, H/8, W/8)
        x = x.mean(dim=(2, 3))                               # (B, w_last)  global average pool
        return self.head(x)                                  # (B, D)


# ================================================================================ HSI
class SpectralAttentionPool(nn.Module):
    """Learned soft crown mask: attention over pixels, used to pool the full-resolution spectrum."""

    def __init__(self, c_feat: int) -> None:
        super().__init__()
        self.score = nn.Conv2d(c_feat, 1, 1)

    def forward(self, feat: torch.Tensor, cube: torch.Tensor) -> torch.Tensor:
        """feat: (B, C, h, w), cube: (B, K, h, w) -> pooled spectrum (B, K)."""
        a = self.score(feat).flatten(1)                      # (B, h*w)
        a = torch.softmax(a, dim=1)                          # (B, h*w)  sums to 1 over pixels
        return torch.einsum("bkn,bn->bk", cube.flatten(2), a)  # (B, K)


class HSIEncoder(nn.Module):
    """Spectral-spatial encoder for hyperspectral crown cubes.

    Two branches:
      1. spatial-spectral: learned band compression (1×1 conv K→c_red) + residual 3×3 conv blocks
         + squeeze-excitation channel attention, global-average pooled;
      2. spectral: attention-pooled crown spectrum (B, K) → 1-D CNN along wavelength (captures
         red-edge position / absorption-feature shapes) → pooled.
    Both are concatenated and projected to D.
    """

    def __init__(self, in_bands: int, embed_dim: int = 128, reduced: int = 64, spatial_width: int = 128,
                 spectral_channels: Sequence[int] = (16, 32, 64), dropout: float = 0.1) -> None:
        super().__init__()
        self.in_bands = in_bands
        self.reduce = nn.Sequential(nn.Conv2d(in_bands, reduced, 1, bias=False), _gn(reduced), nn.GELU())
        self.spatial = nn.Sequential(ResBlock2d(reduced, spatial_width), ResBlock2d(spatial_width, spatial_width))
        self.se = nn.Sequential(nn.Linear(spatial_width, spatial_width // 4), nn.GELU(),
                                nn.Linear(spatial_width // 4, spatial_width), nn.Sigmoid())
        self.attn_pool = SpectralAttentionPool(spatial_width)
        layers: list[nn.Module] = []
        c = 1
        for co in spectral_channels:
            layers += [nn.Conv1d(c, co, 7, stride=2, padding=3, bias=False), _gn(co), nn.GELU()]
            c = co
        self.spectral = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(spatial_width + 2 * c, embed_dim),
                                  nn.LayerNorm(embed_dim))

    def forward(self, hsi: torch.Tensor) -> torch.Tensor:
        """hsi: (B, K, h, w) -> (B, D)."""
        if hsi.shape[1] != self.in_bands:
            raise ValueError(f"expected {self.in_bands} bands, got {hsi.shape[1]}")
        x = self.reduce(hsi)                                 # (B, c_red, h, w)
        x = self.spatial(x)                                  # (B, Cs, h, w)
        g = x.mean(dim=(2, 3))                               # (B, Cs)
        g = g * self.se(g)                                   # (B, Cs)  channel re-weighting
        spec = self.attn_pool(x, hsi)                        # (B, K)   crown spectrum
        s = self.spectral(spec.unsqueeze(1))                 # (B, Cl, K/8)
        s = torch.cat([s.mean(-1), s.amax(-1)], dim=1)       # (B, 2*Cl)
        return self.head(torch.cat([g, s], dim=1))           # (B, D)


# ============================================================================== LiDAR
def masked_quantiles(v: torch.Tensor, mask: torch.Tensor, qs: Sequence[float]) -> torch.Tensor:
    """Per-row quantiles of v over valid entries. v, mask: (B, P) -> (B, len(qs)); 0 for empty rows."""
    big = torch.finfo(v.dtype).max
    vs = torch.where(mask, v, torch.full_like(v, big)).sort(dim=1).values   # (B, P) valid first
    n = mask.sum(1).clamp(min=1)                                            # (B,)
    q = torch.tensor(qs, device=v.device, dtype=v.dtype)                    # (Q,)
    idx = ((n.unsqueeze(1).to(v.dtype) - 1) * q).round().long()             # (B, Q)
    out = vs.gather(1, idx)                                                 # (B, Q)
    return torch.where(mask.any(1, keepdim=True), out, torch.zeros_like(out))  # (B, Q)


class LiDAREncoder(nn.Module):
    """Structural encoder for crown point clouds (permutation-invariant, padding-invariant).

    PointNet-style shared MLP on (x, y, z, intensity) with masked max + mean pooling, plus a
    branch on explicit structural descriptors (height quantiles, vertical/horizontal spread,
    log point count) that encode crown height and shape directly.
    """

    N_STATS: int = 11  # 7 z-quantiles + std(xy) + std(z) + mean intensity + log count

    def __init__(self, in_features: int = 4, embed_dim: int = 128, widths: Sequence[int] = (64, 128, 256),
                 dropout: float = 0.1) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        c = in_features
        for w in widths:
            layers += [nn.Linear(c, w), nn.LayerNorm(w), nn.GELU()]
            c = w
        self.point_mlp = nn.Sequential(*layers)
        self.stats_mlp = nn.Sequential(nn.Linear(self.N_STATS, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, 64))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * c + 64, embed_dim), nn.LayerNorm(embed_dim))

    @staticmethod
    def structural_stats(pts: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """pts: (B, P, 4), mask: (B, P) bool -> (B, 11) masked structural descriptors."""
        m = mask.to(pts.dtype)                                                  # (B, P)
        n = m.sum(1, keepdim=True).clamp(min=1.0)                               # (B, 1)
        zq = masked_quantiles(pts[..., 2], mask, (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0))  # (B, 7)
        mean = (pts * m.unsqueeze(-1)).sum(1) / n                               # (B, 4)
        var = (((pts - mean.unsqueeze(1)) ** 2) * m.unsqueeze(-1)).sum(1) / n   # (B, 4)
        std = var.clamp(min=0).sqrt()                                           # (B, 4)
        sxy = std[:, :2].mean(1, keepdim=True)                                  # (B, 1)
        return torch.cat([zq, sxy, std[:, 2:3], mean[:, 3:4], torch.log1p(m.sum(1, keepdim=True))], 1)  # (B, 11)

    def forward(self, pts: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """pts: (B, P, F), mask: (B, P) bool (True = real point) -> (B, D)."""
        m = mask.unsqueeze(-1)                                                  # (B, P, 1)
        f = self.point_mlp(pts)                                                 # (B, P, C)
        neg = torch.finfo(f.dtype).min
        fmax = torch.where(m, f, torch.full_like(f, neg)).amax(1)               # (B, C)
        has = mask.any(1, keepdim=True)                                         # (B, 1)
        fmax = torch.where(has, fmax, torch.zeros_like(fmax))                   # (B, C) empty -> 0
        fmean = (f * m).sum(1) / m.sum(1).clamp(min=1)                          # (B, C)
        st = self.stats_mlp(self.structural_stats(pts, mask))                   # (B, 64)
        return self.head(torch.cat([fmax, fmean, st], dim=1))                   # (B, D)


# ============================================================================ builder
def build_encoders(mcfg: dict[str, Any], hsi_bands: int, lidar_features: int = 4) -> nn.ModuleDict:
    """Build {'rgb', 'hsi', 'lidar'} encoders from the ``model:`` config section.

    ``hsi_bands`` / ``lidar_features`` come from the data (never hard-coded).
    """
    d, p = int(mcfg["embed_dim"]), float(mcfg.get("dropout", 0.1))
    return nn.ModuleDict({
        "rgb": RGBEncoder(d, mcfg["rgb_widths"], p),
        "hsi": HSIEncoder(hsi_bands, d, mcfg["hsi_reduced"], mcfg["hsi_spatial_width"],
                          mcfg["hsi_spectral_channels"], p),
        "lidar": LiDAREncoder(lidar_features, d, mcfg["lidar_widths"], p),
    })


def count_parameters(module: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
