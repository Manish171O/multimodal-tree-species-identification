"""Site-held-out split: sites never overlap; every sample lands in exactly one split."""
from __future__ import annotations

import numpy as np
import pytest

from src.data.dataset import build_split
from src.data.splits import site_held_out_split


def _toy_sites(n: int = 1000, n_sites: int = 10, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    sites = np.array([f"S{k:02d}".encode() for k in rng.integers(0, n_sites, n)])
    return sites, rng.integers(0, 30, n)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_no_site_overlap_and_full_coverage(seed: int) -> None:
    sites, labels = _toy_sites(seed=seed)
    sp = site_held_out_split(sites, labels, test_frac=0.2, val_frac=0.15, seed=seed)
    tr, va, te = set(sp.train_sites), set(sp.val_sites), set(sp.test_sites)
    assert not (tr & va) and not (tr & te) and not (va & te), "site overlap between splits"
    assert tr and va and te
    dec = np.array([s.decode() for s in sites])
    # sample-level: indices disjoint, cover everything, and each index's site belongs to its split
    all_idx = np.concatenate([sp.train_idx, sp.val_idx, sp.test_idx])
    assert len(all_idx) == len(set(all_idx.tolist())) == len(sites)
    assert set(dec[sp.train_idx]) == tr and set(dec[sp.val_idx]) == va and set(dec[sp.test_idx]) == te
    assert len(sp.test_idx) >= 0.2 * len(sites) and len(sp.val_idx) >= 0.15 * len(sites)


def test_deterministic_given_seed() -> None:
    sites, _ = _toy_sites()
    a = site_held_out_split(sites, seed=7)
    b = site_held_out_split(sites, seed=7)
    assert a.test_sites == b.test_sites and np.array_equal(a.train_idx, b.train_idx)


def test_explicit_sites_honoured() -> None:
    sites, _ = _toy_sites()
    sp = site_held_out_split(sites, test_sites=["S01", "S02"], val_sites=["S03"])
    assert sp.test_sites == ["S01", "S02"] and sp.val_sites == ["S03"]
    assert "S01" not in sp.train_sites


@pytest.mark.parametrize("kw", [dict(test_sites=["NOPE"]), dict(test_sites=["S01"], val_sites=["S01"])])
def test_bad_config_raises(kw: dict) -> None:
    sites, _ = _toy_sites()
    with pytest.raises(ValueError):
        site_held_out_split(sites, **kw)


def test_too_few_sites_raises() -> None:
    with pytest.raises(ValueError):
        site_held_out_split(np.array(["A", "A", "B"]))


def test_split_on_synthetic_file(cfg: dict) -> None:
    sp = build_split(cfg)
    assert not set(sp.train_sites) & set(sp.test_sites)
    assert not set(sp.train_sites) & set(sp.val_sites)
    assert not set(sp.val_sites) & set(sp.test_sites)
    assert "test_species_unseen_in_train" in sp.info
