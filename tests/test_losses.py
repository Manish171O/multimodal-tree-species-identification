"""Evidential and cross-modal contrastive losses."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from src.losses import cross_modal_contrastive_loss, evidential_loss, kl_dirichlet_uniform, pairwise_infonce

MODS = ("rgb", "hsi", "lidar")


# ------------------------------------------------------------------ evidential
def test_kl_zero_for_uniform_dirichlet() -> None:
    assert torch.allclose(kl_dirichlet_uniform(torch.ones(4, 7)), torch.zeros(4), atol=1e-5)
    assert (kl_dirichlet_uniform(torch.rand(4, 7) * 10 + 1.5) > 0).all()


def test_evidential_rewards_correct_evidence() -> None:
    y = torch.tensor([2, 0])
    good = torch.ones(2, 1, 5)
    good[0, 0, 2] = good[1, 0, 0] = 50.0
    bad = torch.ones(2, 1, 5)
    bad[0, 0, 3] = bad[1, 0, 4] = 50.0
    m = torch.ones(2, 1)
    assert evidential_loss(good, y, m, 0.1) < evidential_loss(torch.ones(2, 1, 5), y, m, 0.1) < evidential_loss(bad, y, m, 0.1)


def test_evidential_ignores_missing_modalities() -> None:
    y = torch.tensor([1])
    alpha = torch.ones(1, 2, 4)
    alpha[0, 0, 1] = 20.0
    alpha_garbage = alpha.clone()
    alpha_garbage[0, 1] = torch.tensor([1e3, 1.0, 1e3, 1e3])
    m = torch.tensor([[1.0, 0.0]])
    assert torch.allclose(evidential_loss(alpha, y, m), evidential_loss(alpha_garbage, y, m))
    assert evidential_loss(alpha, y, torch.zeros(1, 2)) == 0


def test_evidential_gradient_finite() -> None:
    logits = torch.randn(8, 3, 10, requires_grad=True)
    loss = evidential_loss(F.softplus(logits) + 1, torch.randint(0, 10, (8,)), torch.ones(8, 3), 0.1)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()


# ----------------------------------------------------------------- contrastive
def test_infonce_aligned_lower_than_random() -> None:
    torch.manual_seed(0)
    z = F.normalize(torch.randn(16, 32), dim=-1)
    aligned = pairwise_infonce(z, z, 0.1)
    rand = pairwise_infonce(z, F.normalize(torch.randn(16, 32), dim=-1), 0.1)
    assert aligned < 0.1 and rand > aligned


def test_infonce_uniform_logits_equals_log_n() -> None:
    z = torch.zeros(8, 4)
    assert math.isclose(pairwise_infonce(z, z, 0.1).item(), math.log(8), rel_tol=1e-5)


def test_infonce_same_class_exclusion() -> None:
    za = F.normalize(torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]), dim=-1)
    with_neg = pairwise_infonce(za, za, 0.1)
    excl = pairwise_infonce(za, za, 0.1, labels=torch.tensor([0, 0, 1]))
    assert excl < with_neg                                    # near-duplicate same-species pair no longer penalised


def test_contrastive_uses_only_valid_pairs() -> None:
    torch.manual_seed(0)
    proj = {k: F.normalize(torch.randn(6, 8), dim=-1) for k in MODS}
    mask = torch.ones(6, 3)
    mask[:3, 2] = 0                                            # LiDAR missing for 3 trees
    base = cross_modal_contrastive_loss(proj, mask, MODS)
    proj2 = dict(proj)
    l2 = proj["lidar"].clone()
    l2[:3] = F.normalize(torch.randn(3, 8), dim=-1)            # change only the missing rows
    proj2["lidar"] = l2
    assert torch.allclose(base, cross_modal_contrastive_loss(proj2, mask, MODS))


def test_contrastive_no_valid_pairs_is_zero_with_grad() -> None:
    proj = {k: F.normalize(torch.randn(4, 8), dim=-1).requires_grad_() for k in MODS}
    mask = torch.zeros(4, 3)
    mask[:, 0] = 1                                             # only RGB present -> no pairs
    loss = cross_modal_contrastive_loss(proj, mask, MODS)
    loss.backward()
    assert loss.item() == 0.0
