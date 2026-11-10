"""Encoder output shapes, missing-input safety, LiDAR mask/permutation invariance, gradients."""
from __future__ import annotations

import pytest
import torch

from src.data.dataset import build_dataloaders
from src.models.encoders import (HSIEncoder, LiDAREncoder, RGBEncoder, build_encoders,
                                 count_parameters, masked_quantiles)

D = 128


@pytest.fixture()
def encoders(cfg) -> torch.nn.ModuleDict:
    torch.manual_seed(0)
    return build_encoders(cfg["model"], hsi_bands=369).eval()


@pytest.mark.parametrize("b", [1, 4])
def test_rgb_shape(b: int) -> None:
    out = RGBEncoder(D).eval()(torch.randn(b, 3, 64, 64))
    assert out.shape == (b, D) and torch.isfinite(out).all()


@pytest.mark.parametrize("b,hw", [(1, 12), (4, 12), (2, 8)])
def test_hsi_shape(b: int, hw: int) -> None:
    out = HSIEncoder(369, D).eval()(torch.randn(b, 369, hw, hw))
    assert out.shape == (b, D) and torch.isfinite(out).all()


def test_hsi_rejects_wrong_band_count() -> None:
    with pytest.raises(ValueError):
        HSIEncoder(369, D)(torch.randn(1, 368, 12, 12))


@pytest.mark.parametrize("b,p", [(1, 512), (4, 512), (3, 100)])
def test_lidar_shape(b: int, p: int) -> None:
    mask = torch.rand(b, p) > 0.3
    out = LiDAREncoder(4, D).eval()(torch.randn(b, p, 4), mask)
    assert out.shape == (b, D) and torch.isfinite(out).all()


def test_all_encoders_same_dim(encoders) -> None:
    x = {"rgb": torch.randn(2, 3, 64, 64), "hsi": torch.randn(2, 369, 12, 12)}
    outs = [encoders["rgb"](x["rgb"]), encoders["hsi"](x["hsi"]),
            encoders["lidar"](torch.randn(2, 512, 4), torch.ones(2, 512, dtype=torch.bool))]
    assert {o.shape for o in outs} == {torch.Size([2, D])}


def test_missing_inputs_give_finite_embeddings(encoders) -> None:
    """Missing sensors arrive as zeros (and an all-False LiDAR mask) -> no NaN/inf."""
    outs = [encoders["rgb"](torch.zeros(2, 3, 64, 64)), encoders["hsi"](torch.zeros(2, 369, 12, 12)),
            encoders["lidar"](torch.zeros(2, 512, 4), torch.zeros(2, 512, dtype=torch.bool))]
    for o in outs:
        assert torch.isfinite(o).all()


def test_lidar_ignores_padding_values() -> None:
    enc = LiDAREncoder(4, D).eval()
    pts, mask = torch.randn(2, 64, 4), torch.zeros(2, 64, dtype=torch.bool)
    mask[:, :40] = True
    a = enc(pts, mask)
    pts2 = pts.clone()
    pts2[~mask] = 1e3 * torch.randn_like(pts2[~mask])      # garbage in padded rows
    assert torch.allclose(a, enc(pts2, mask), atol=1e-5)


def test_lidar_permutation_invariant() -> None:
    enc = LiDAREncoder(4, D).eval()
    pts, mask = torch.randn(2, 64, 4), torch.rand(2, 64) > 0.2
    perm = torch.randperm(64)
    assert torch.allclose(enc(pts, mask), enc(pts[:, perm], mask[:, perm]), atol=1e-5)


def test_masked_quantiles_match_torch() -> None:
    v = torch.randn(3, 50)
    mask = torch.ones(3, 50, dtype=torch.bool)
    mask[1, 25:] = False
    mask[2] = False
    q = masked_quantiles(v, mask, (0.0, 0.5, 1.0))
    assert torch.allclose(q[0, [0, 2]], torch.stack([v[0].min(), v[0].max()]))
    assert torch.allclose(q[1, 2], v[1, :25].max()) and (q[2] == 0).all()


def test_gradients_reach_every_parameter(encoders) -> None:
    encoders.train()
    loss = (encoders["rgb"](torch.randn(2, 3, 64, 64)).sum() + encoders["hsi"](torch.randn(2, 369, 12, 12)).sum()
            + encoders["lidar"](torch.randn(2, 128, 4), torch.rand(2, 128) > 0.2).sum())
    loss.backward()
    missing = [n for n, p in encoders.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    assert not missing, f"no/invalid grad for {missing}"


def test_encoders_on_real_batch(cfg, encoders) -> None:
    batch = next(iter(build_dataloaders(cfg)[0]))
    bs = batch["rgb"].shape[0]
    with torch.no_grad():
        e = [encoders["rgb"](batch["rgb"]), encoders["hsi"](batch["hsi"]),
             encoders["lidar"](batch["lidar"], batch["lidar_mask"])]
    for o in e:
        assert o.shape == (bs, cfg["model"]["embed_dim"]) and torch.isfinite(o).all()
    assert all(count_parameters(m) > 0 for m in encoders.values())
