"""Site-held-out train/val/test splitting. Whole sites go to exactly one split — never random rows."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SiteSplit:
    """Indices and site lists for each split."""

    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    train_sites: list[str]
    val_sites: list[str]
    test_sites: list[str]
    info: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        """JSON-serialisable summary (sites, counts, species coverage)."""
        return {"train_sites": self.train_sites, "val_sites": self.val_sites, "test_sites": self.test_sites,
                "n_train": int(len(self.train_idx)), "n_val": int(len(self.val_idx)),
                "n_test": int(len(self.test_idx)), **self.info}


def _pick_sites(pool: list[str], counts: dict[str, int], target: int, rng: np.random.Generator) -> list[str]:
    """Randomly add sites from ``pool`` until their total sample count reaches ``target``."""
    chosen: list[str] = []
    total = 0
    for s in rng.permutation(pool):
        if total >= target:
            break
        chosen.append(str(s))
        total += counts[str(s)]
    return chosen


def site_held_out_split(
    sites: np.ndarray,
    labels: np.ndarray | None = None,
    test_sites: list[str] | None = None,
    val_sites: list[str] | None = None,
    test_frac: float = 0.2,
    val_frac: float = 0.15,
    seed: int = 42,
) -> SiteSplit:
    """Split sample indices by site.

    Args:
        sites: (N,) array of site codes (str or bytes).
        labels: optional (N,) species ids, used to report species coverage.
        test_sites / val_sites: explicit site lists; if empty, sites are drawn at random
            (seeded) until each split holds at least ``*_frac`` of samples.
        test_frac, val_frac: target sample fractions for automatic selection.
        seed: RNG seed for automatic selection.

    Returns:
        SiteSplit with disjoint site sets and the matching sample indices.
    """
    sites = np.array([s.decode() if isinstance(s, bytes) else str(s) for s in sites])
    uniq, cnt = np.unique(sites, return_counts=True)
    counts = dict(zip(uniq.tolist(), cnt.tolist()))
    if len(uniq) < 3:
        raise ValueError(f"need >= 3 sites for a train/val/test site split, got {len(uniq)}")
    rng = np.random.default_rng(seed)
    n = len(sites)

    test_sites = list(test_sites or [])
    val_sites = list(val_sites or [])
    unknown = set(test_sites + val_sites) - set(counts)
    if unknown:
        raise ValueError(f"unknown sites in config: {sorted(unknown)}")
    if set(test_sites) & set(val_sites):
        raise ValueError("a site cannot be in both val_sites and test_sites")
    if not test_sites:
        pool = [s for s in uniq.tolist() if s not in val_sites]
        test_sites = _pick_sites(pool, counts, int(np.ceil(test_frac * n)), rng)
    if not val_sites:
        pool = [s for s in uniq.tolist() if s not in test_sites]
        val_sites = _pick_sites(pool, counts, int(np.ceil(val_frac * n)), rng)
    train_sites = [s for s in uniq.tolist() if s not in test_sites and s not in val_sites]
    if not train_sites:
        raise ValueError("no sites left for training")

    idx = np.arange(n)
    split = SiteSplit(
        train_idx=idx[np.isin(sites, train_sites)],
        val_idx=idx[np.isin(sites, val_sites)],
        test_idx=idx[np.isin(sites, test_sites)],
        train_sites=sorted(train_sites), val_sites=sorted(val_sites), test_sites=sorted(test_sites),
    )
    if labels is not None:
        tr = set(np.unique(labels[split.train_idx]).tolist())
        for name, ix in (("val", split.val_idx), ("test", split.test_idx)):
            sp = set(np.unique(labels[ix]).tolist())
            split.info[f"{name}_species"] = len(sp)
            split.info[f"{name}_species_unseen_in_train"] = len(sp - tr)
            split.info[f"{name}_samples_unseen_species"] = int(np.isin(labels[ix], list(sp - tr)).sum())
        split.info["train_species"] = len(tr)
    return split
