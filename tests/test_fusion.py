"""Quality estimator, adaptive fusion with missing modalities, heads and full-model forward."""
from __future__ import annotations

from itertools import product

import pytest
import torch

from src.data.dataset import build_dataloaders, build_datasets
from src.models.fusion import AdaptiveFusion
from src.models.model import apply_modality_mask, build_model
from src.models.quality import QualityEstimator

B, M, D, C = 6, 3, 16, 10
SUBSETS = [torch.tensor(s, dtype=torch.float32) for s in product([0, 1], repeat=3) if any(s)]


def test_quality_estimator_ranges_and_missing() -> None:
    qe = QualityEstimator(("rgb", "hsi", "lidar"), D, C)
    embs = {k: torch.randn(B, D) for k in ("rgb", "hsi", "lidar")}
    mask = torch.ones(B, M)
    mask[0, 1] = 0
    q = qe(embs, mask)
    assert q["alpha"].shape == (B, M, C) and (q["alpha"] >= 1).all()
    assert q["trust"].shape == (B, M)
    assert ((q["trust"] >= 0) & (q["trust"] < 1)).all()
    assert q["trust"][0, 1] == 0 and q["uncertainty"][0, 1] == 1
    assert torch.allclose(q["uncertainty"][mask > 0] + q["trust"][mask > 0], torch.ones(int(mask.sum())))


def test_trust_increases_with_evidence() -> None:
    qe = QualityEstimator(("a",), D, C)
    with torch.no_grad():
        qe.heads["a"].net[-1].bias.fill_(-10.0)          # ~zero evidence
    low = qe({"a": torch.zeros(1, D)}, torch.ones(1, 1))["trust"]
    with torch.no_grad():
        qe.heads["a"].net[-1].bias.fill_(10.0)           # lots of evidence
    high = qe({"a": torch.zeros(1, D)}, torch.ones(1, 1))["trust"]
    assert low.item() < 0.01 and high.item() > 0.9


@pytest.mark.parametrize("mode", ["trust", "mean"])
@pytest.mark.parametrize("keep", SUBSETS, ids=lambda k: "".join(str(int(v)) for v in k))
def test_fusion_missing_modalities(mode: str, keep: torch.Tensor) -> None:
    fus = AdaptiveFusion(D, mode=mode).eval()
    embs, trust = torch.randn(B, M, D), torch.rand(B, M)
    mask = keep.expand(B, M).clone()
    fused, w = fus(embs, mask, trust)
    assert fused.shape == (B, D) and torch.isfinite(fused).all()
    assert (w[:, keep == 0] == 0).all(), "missing modality got non-zero weight"
    assert torch.allclose(w.sum(1), torch.ones(B))
    # garbage in a missing modality's embedding must not change the output
    embs2 = embs.clone()
    embs2[:, keep == 0] = 1e4 * torch.randn_like(embs2[:, keep == 0])
    assert torch.allclose(fused, fus(embs2, mask, trust)[0], atol=1e-5)


def test_fusion_all_missing_is_zero_and_finite() -> None:
    fused, w = AdaptiveFusion(D)(torch.randn(B, M, D), torch.zeros(B, M), torch.rand(B, M))
    assert (fused == 0).all() and (w == 0).all()


def test_trust_weighting_prefers_trusted_modality() -> None:
    fus = AdaptiveFusion(D, gamma=1.0)
    w = fus.weights(torch.ones(1, 3), torch.tensor([[0.9, 0.1, 0.0]]))
    assert w[0, 0] > w[0, 1] > w[0, 2] and torch.isclose(w.sum(), torch.tensor(1.0))


def test_detach_trust_blocks_gradient() -> None:
    trust = torch.rand(B, M, requires_grad=True)
    fused, _ = AdaptiveFusion(D, detach_trust=True)(torch.randn(B, M, D), torch.ones(B, M), trust)
    fused.sum().backward()
    assert trust.grad is None
    trust2 = torch.rand(B, M, requires_grad=True)
    fused, _ = AdaptiveFusion(D, detach_trust=False)(torch.randn(B, M, D), torch.ones(B, M), trust2)
    fused.sum().backward()
    assert trust2.grad is not None


@pytest.fixture()
def model_and_batch(cfg):
    torch.manual_seed(0)
    tr, _, _, _ = build_datasets(cfg)
    batch = next(iter(build_dataloaders(cfg)[1]))  # val loader: deterministic
    return build_model(cfg, tr).eval(), batch, tr


def test_full_model_output_shapes(model_and_batch, cfg) -> None:
    model, batch, tr = model_and_batch
    bs, d, c = batch["rgb"].shape[0], cfg["model"]["embed_dim"], tr.num_classes
    with torch.no_grad():
        out = model(batch)
    assert out["logits"].shape == (bs, c) and out["structure"].shape == (bs, 2)
    assert out["embeddings"].shape == (bs, 3, d) and out["alpha"].shape == (bs, 3, c)
    assert out["weights"].shape == out["trust"].shape == (bs, 3)
    assert all(v.shape == (bs, cfg["model"]["proj_dim"]) for v in out["proj"].values())
    assert model.structure_metres(out).shape == (bs, 2)
    for k in ("logits", "structure", "fused", "trust", "weights"):
        assert torch.isfinite(out[k]).all()


@pytest.mark.parametrize("keep", SUBSETS, ids=lambda k: "".join(str(int(v)) for v in k))
def test_full_model_with_dropped_sensors(model_and_batch, keep: torch.Tensor) -> None:
    model, batch, _ = model_and_batch
    bs = batch["rgb"].shape[0]
    k = keep.expand(bs, 3)
    with torch.no_grad():
        out = model(batch, keep=k)
        # Changing the inputs of dropped sensors must not change predictions.
        noisy = dict(batch)
        for i, name in enumerate(("rgb", "hsi", "lidar")):
            if keep[i] == 0:
                noisy[name] = batch[name] + 5 * torch.randn_like(batch[name])
        out2 = model(noisy, keep=k)
    assert torch.isfinite(out["logits"]).all()
    assert (out["weights"][:, keep == 0] == 0).all()
    assert torch.allclose(out["logits"], out2["logits"], atol=1e-4)


def test_apply_modality_mask_keeps_existing_missing(model_and_batch) -> None:
    _, batch, _ = model_and_batch
    b = apply_modality_mask(batch, torch.ones(batch["rgb"].shape[0], 3))
    assert torch.equal(b["modality_mask"], batch["modality_mask"])
    b = apply_modality_mask(batch, torch.tensor([[0.0, 1.0, 1.0]]).expand(batch["rgb"].shape[0], 3))
    assert (b["modality_mask"][:, 0] == 0).all() and (b["rgb"] == 0).all()


def test_mean_fusion_ablation_builds(cfg) -> None:
    tr, _, _, _ = build_datasets(cfg)
    cfg["model"]["use_quality_estimator"] = False
    model = build_model(cfg, tr).eval()
    batch = next(iter(build_dataloaders(cfg)[1]))
    with torch.no_grad():
        out = model(batch)
    present = batch["modality_mask"]
    assert torch.allclose(out["weights"], present / present.sum(1, keepdim=True))


def test_evidence_activations() -> None:
    from src.models.quality import EvidentialHead
    for act in ("exp", "softplus"):
        a = EvidentialHead(D, C, activation=act)(torch.randn(4, D) * 100)   # large inputs: clamp keeps exp finite
        assert a.shape == (4, C) and (a >= 1).all() and torch.isfinite(a).all()
    with pytest.raises(ValueError):
        EvidentialHead(D, C, activation="relu6")
