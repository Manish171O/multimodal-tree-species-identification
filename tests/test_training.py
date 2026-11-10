"""Focal loss, class-balanced weights/sampler, modality dropout, total loss, LR schedule."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.losses import FocalLoss, TotalLoss, class_balanced_weights, kl_anneal
from src.training.modality_dropout import sample_modality_keep
from src.training.sampler import class_balanced_sampler
from src.training.trainer import warmup_cosine


def test_focal_gamma0_equals_cross_entropy() -> None:
    logits, y = torch.randn(16, 7), torch.randint(0, 7, (16,))
    assert torch.allclose(FocalLoss(0.0)(logits, y), F.cross_entropy(logits, y), atol=1e-6)


def test_focal_downweights_easy_examples() -> None:
    y = torch.tensor([0])
    easy, hard = torch.tensor([[5.0, 0.0, 0.0]]), torch.tensor([[0.2, 0.0, 0.0]])
    ratio_easy = FocalLoss(2.0)(easy, y) / F.cross_entropy(easy, y)
    ratio_hard = FocalLoss(2.0)(hard, y) / F.cross_entropy(hard, y)
    assert ratio_easy < ratio_hard < 1


def test_focal_alpha_weights_classes() -> None:
    logits, y = torch.zeros(2, 2), torch.tensor([0, 1])
    logits[0, 1] = 3.0                                    # sample of class 0 is badly wrong
    lo = FocalLoss(0.0, torch.tensor([0.1, 1.0]))(logits, y)
    hi = FocalLoss(0.0, torch.tensor([1.0, 0.1]))(logits, y)
    assert hi > lo


def test_class_balanced_weights_favour_rare() -> None:
    w = class_balanced_weights(np.array([1000, 10, 1, 0]))
    assert w[2] > w[1] > w[0] and w[3] == 0 and torch.isclose(w[:3].mean(), torch.tensor(1.0))


@pytest.mark.parametrize("power,max_ratio", [(1.0, 1.5), (0.0, None)])
def test_sampler_balances(power: float, max_ratio: float | None) -> None:
    labels = np.array([0] * 900 + [1] * 90 + [2] * 10)
    idx = np.array(list(class_balanced_sampler(labels, power, num_samples=30000, seed=0)))
    freq = np.bincount(labels[idx], minlength=3) / len(idx)
    if max_ratio:
        assert freq.max() / freq.min() < max_ratio        # roughly uniform over classes
    else:
        assert np.allclose(freq, [0.9, 0.09, 0.01], atol=0.01)


def test_sampler_reproducible() -> None:
    labels = np.random.default_rng(0).integers(0, 5, 100)
    assert list(class_balanced_sampler(labels, seed=3)) == list(class_balanced_sampler(labels, seed=3))


def test_modality_dropout_never_removes_all_and_respects_missing() -> None:
    g = torch.Generator().manual_seed(0)
    mask = torch.ones(5000, 3)
    mask[:1000, 0] = 0                                    # RGB already missing for some trees
    mask[1000:1500, :2] = 0                               # only LiDAR present
    keep = sample_modality_keep(mask, 1.0, (0.9, 0.9, 0.9), g)
    eff = keep * mask
    assert (eff.sum(1) >= 1).all(), "a sample lost every sensor"
    assert (eff[1000:1500, 2] == 1).all(), "the only present sensor was dropped"
    assert set(keep.unique().tolist()) <= {0.0, 1.0}
    rate = 1 - keep[1500:].mean(0)
    assert (rate > 0.6).all()                              # dropout is really applied


def test_modality_dropout_p_apply_zero_keeps_everything() -> None:
    keep = sample_modality_keep(torch.ones(100, 3), 0.0, (1.0, 1.0, 1.0), torch.Generator().manual_seed(0))
    assert (keep == 1).all()


def test_kl_anneal_and_lr_schedule() -> None:
    assert kl_anneal(0, 10, 0.1) == 0 and kl_anneal(5, 10, 0.1) == pytest.approx(0.05) and kl_anneal(50, 10, 0.1) == 0.1
    assert warmup_cosine(0, 100, 10) == pytest.approx(0.1) and warmup_cosine(9, 100, 10) == pytest.approx(1.0)
    assert warmup_cosine(100, 100, 10) == pytest.approx(0.01)


def test_total_loss_components(cfg) -> None:
    b, c, mods = 8, 5, ("rgb", "hsi", "lidar")
    out = {"logits": torch.randn(b, c, requires_grad=True), "structure": torch.randn(b, 2, requires_grad=True),
           "proj": {m: F.normalize(torch.randn(b, 16), dim=-1).requires_grad_() for m in mods},
           "alpha": (F.softplus(torch.randn(b, 3, c)) + 1).requires_grad_(), "modality_mask": torch.ones(b, 3)}
    batch = {"label": torch.randint(0, c, (b,)), "height": torch.rand(b) * 30, "crown_diameter": torch.rand(b) * 8}
    crit = TotalLoss(cfg["loss"], mods)
    res = crit(out, batch, torch.tensor([15.0, 4.0]), torch.tensor([5.0, 2.0]), epoch=3)
    for k in ("total", "cls", "contrastive", "aux", "evidential"):
        assert torch.isfinite(res[k])
    lc = cfg["loss"]
    expect = (lc["w_cls"] * res["cls"] + lc["w_contrastive"] * res["contrastive"] + lc["w_aux"] * res["aux"]
              + lc["w_evidential"] * res["evidential"])
    assert torch.isclose(res["total"], expect)
    res["total"].backward()
    assert out["logits"].grad is not None and out["alpha"].grad is not None
    w_con = lc["w_contrastive"]
    cfg["loss"]["w_contrastive"] = 0.0                     # ablation switch removes the term
    r0 = TotalLoss(cfg["loss"], mods)(out, batch, torch.tensor([15.0, 4.0]), torch.tensor([5.0, 2.0]), 3)
    assert torch.isclose(r0["total"], expect - w_con * res["contrastive"])
