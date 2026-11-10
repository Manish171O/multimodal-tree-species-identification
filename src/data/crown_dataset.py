"""Loader for the compact `demo_tree_crowns.h5` layout (also written by scripts/make_real_species_data.py).

Layout (verified against the reference file): rgb (N,3,32,32) uint8 · hsi (N,369,8,8) float16 reflectance ·
lidar (N,1,32,32) float32 canopy-height raster · species/site int64 · height/crown_diameter float32 ·
degraded (N,3) bool · wavelengths (369,) · attrs species_names / site_names.

The LiDAR height raster is converted to a point set so the existing permutation-invariant LiDAR
encoder is reused unchanged: every pixel with a return becomes a point (x, y, z, slope), where x, y are
metres from the window centre, z is canopy height and slope is the local height-gradient magnitude.
A value of exactly 0.0 marks a pixel with no return (sparse / degraded LiDAR in this layout).

Items have exactly the same keys and shapes contract as `TreeCrownDataset`, so the model, trainer,
evaluation suite and predict.py work on both layouts.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.dataset import brightness_normalize, spectral_augment
from src.data.splits import SiteSplit, site_held_out_split

FORMAT = "demo_crowns"
STATS_VERSION = 1
REQUIRED = {"rgb": 4, "hsi": 4, "lidar": 4, "species": 1, "site": 1, "height": 1, "crown_diameter": 1}


_ARRAYS: dict[str, dict[str, np.ndarray]] = {}


def load_arrays(path: str | Path) -> dict[str, np.ndarray]:
    """Whole rgb/hsi/lidar arrays, read once per process and cached (the compact layout is ~0.1 GB per
    2000 crowns, and its gzip chunks span 125 crowns, so per-crown reads would be very slow)."""
    key = str(Path(path).resolve())
    if key not in _ARRAYS:
        with h5py.File(path, "r") as f:
            _ARRAYS[key] = {"rgb": f["rgb"][:], "hsi": f["hsi"][:], "lidar": f["lidar"][:]}
    return _ARRAYS[key]


def _attr_list(f: h5py.File, key: str) -> list[str] | None:
    if key not in f.attrs:
        return None
    return [v.decode() if isinstance(v, bytes) else str(v) for v in f.attrs[key]]


def read_meta(path: str | Path) -> dict[str, Any]:
    """Labels and names from the file (checks the schema first; shapes are read, never assumed)."""
    with h5py.File(path, "r") as f:
        for k, rank in REQUIRED.items():
            if k not in f or len(f[k].shape) != rank:
                raise ValueError(f"{path}: expected key '{k}' with rank {rank} (demo_crowns layout)")
        n_cls = int(f["species"][:].max()) + 1
        names = _attr_list(f, "species_names") or [f"species_{i:02d}" for i in range(n_cls)]
        sites = _attr_list(f, "site_names") or [f"SITE{i}" for i in range(int(f["site"][:].max()) + 1)]
        px = f.attrs["pixel_size_m"].tolist() if "pixel_size_m" in f.attrs else [0.5, 2.0, 0.5]
        return {"labels": f["species"][:].astype(np.int64), "site": f["site"][:].astype(np.int64),
                "height": f["height"][:].astype(np.float32), "crown": f["crown_diameter"][:].astype(np.float32),
                "degraded": f["degraded"][:] if "degraded" in f else None,
                "names": names, "scientific": _attr_list(f, "species_scientific_names") or names,
                "codes": _attr_list(f, "species_codes") or names, "site_names": sites, "pixel_size_m": px,
                "synthetic": "SYNTHETIC" in str(f.attrs.get("note", "")).upper() or bool(f.attrs.get("synthetic", False)),
                "rgb_shape": f["rgb"].shape[1:], "hsi_shape": f["hsi"].shape[1:], "lidar_shape": f["lidar"].shape[1:]}


def chm_to_points(chm: np.ndarray, pixel_m: float) -> np.ndarray:
    """Canopy-height raster (H, W) -> points (K, 4) = x, y [m from centre], z [m], slope; 0.0 = no return."""
    chm = np.nan_to_num(chm.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    gy, gx = np.gradient(chm, pixel_m)                                         # (H, W) each
    slope = np.sqrt(gx ** 2 + gy ** 2)                                         # (H, W)
    h, w = chm.shape
    yy, xx = np.meshgrid((np.arange(h) + 0.5 - h / 2) * pixel_m, (np.arange(w) + 0.5 - w / 2) * pixel_m,
                         indexing="ij")
    ok = chm != 0.0
    return np.stack([xx[ok], yy[ok], chm[ok], slope[ok]], 1).astype(np.float32)   # (K, 4)


def stratified_split(labels: np.ndarray, sites: np.ndarray, site_names: list[str], val_frac: float,
                     test_frac: float, seed: int) -> SiteSplit:
    """Per-species random split (every species with >= 3 crowns appears in train, val and test).

    Used when species are confined to single sites (a site-held-out split would make them unlearnable).
    """
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for c in np.unique(labels):
        idx = rng.permutation(np.where(labels == c)[0])
        n = len(idx)
        n_te = max(1, int(round(test_frac * n))) if n >= 3 else 0
        n_va = max(1, int(round(val_frac * n))) if n >= 3 else 0
        te += idx[:n_te].tolist(); va += idx[n_te:n_te + n_va].tolist(); tr += idx[n_te + n_va:].tolist()
    all_sites = sorted({site_names[s] for s in np.unique(sites)})
    return SiteSplit(np.sort(np.array(tr)), np.sort(np.array(va)), np.sort(np.array(te)),
                     all_sites, all_sites, all_sites, {"split_mode": "stratified_by_species"})


def build_demo_split(cfg: dict[str, Any], meta: dict[str, Any] | None = None) -> SiteSplit:
    """Split according to ``split.mode``: 'site' (whole sites held out) or 'stratified'."""
    meta = meta or read_meta(cfg["paths"]["data_file"])
    sc = cfg["split"]
    seed = int(sc["seed"]) if sc.get("seed") is not None else cfg["seed"]
    if sc.get("mode", "site") == "stratified":
        return stratified_split(meta["labels"], meta["site"], meta["site_names"], sc["val_frac"], sc["test_frac"], seed)
    names = np.array([meta["site_names"][s] for s in meta["site"]])
    sp = site_held_out_split(names, meta["labels"], sc.get("test_sites"), sc.get("val_sites"),
                             sc["test_frac"], sc["val_frac"], seed)
    sp.info["split_mode"] = "site_held_out"
    return sp


def compute_demo_stats(path: str | Path, idx: np.ndarray, dcfg: dict[str, Any], meta: dict[str, Any],
                       max_samples: int = 1500, seed: int = 0) -> dict[str, Any]:
    """Per-modality normalization fit on TRAIN indices only, plus class/site names for predict.py."""
    rng = np.random.default_rng(seed)
    sel = np.sort(rng.choice(idx, min(max_samples, len(idx)), replace=False))
    px = float(meta["pixel_size_m"][2])
    arr = load_arrays(path)
    rgb = arr["rgb"][sel].astype(np.float64) / 255.0                                     # (n, 3, H, W)
    hsi = arr["hsi"][sel].astype(np.float64)                                             # (n, B, h, w)
    if dcfg.get("hsi_brightness_norm", False):
        hsi = np.stack([brightness_normalize(x, np.ones(x.shape[1:], bool)) for x in hsi])
    pts = np.concatenate([chm_to_points(c[0], px) for c in arr["lidar"][sel]])          # (K, 4)
    tr_h, tr_c = meta["height"][idx], meta["crown"][idx]
    return {
        "format": FORMAT, "version": STATS_VERSION,
        "rgb_mean": rgb.mean((0, 2, 3)).tolist(), "rgb_std": (rgb.std((0, 2, 3)) + 1e-6).tolist(),
        "hsi_mean": hsi.mean((0, 2, 3)).tolist(), "hsi_std": (hsi.std((0, 2, 3)) + 1e-6).tolist(),
        "lidar_xy_std": float(pts[:, :2].std() + 1e-6), "lidar_z_mean": float(pts[:, 2].mean()),
        "lidar_z_std": float(pts[:, 2].std() + 1e-6), "lidar_i_mean": float(pts[:, 3].mean()),
        "lidar_i_std": float(pts[:, 3].std() + 1e-6),
        "height_mean": float(tr_h.mean()), "height_std": float(tr_h.std() + 1e-6),
        "crown_mean": float(tr_c.mean()), "crown_std": float(tr_c.std() + 1e-6),
        "class_names": meta["names"], "class_scientific": meta["scientific"], "class_codes": meta["codes"],
        "site_names": meta["site_names"], "pixel_size_m": meta["pixel_size_m"],
        "rgb_shape": list(meta["rgb_shape"]), "hsi_shape": list(meta["hsi_shape"]),
        "lidar_shape": list(meta["lidar_shape"]), "hsi_brightness_norm": bool(dcfg.get("hsi_brightness_norm", False)),
        "n_samples_used": int(len(sel)),
    }


class CrownPreprocessor:
    """Turns raw arrays (as stored in the file, or supplied by a user) into model-ready tensors."""

    def __init__(self, stats: dict[str, Any], dcfg: dict[str, Any]) -> None:
        self.s, self.dcfg = stats, dcfg
        self.n_points = int(dcfg["lidar_num_points"])
        self.rgb_mean = np.asarray(stats["rgb_mean"], np.float32)[:, None, None]
        self.rgb_std = np.asarray(stats["rgb_std"], np.float32)[:, None, None]
        self.hsi_mean = np.asarray(stats["hsi_mean"], np.float32)[:, None, None]
        self.hsi_std = np.asarray(stats["hsi_std"], np.float32)[:, None, None]
        self.px = float(stats["pixel_size_m"][2])

    def __call__(self, rgb: np.ndarray | None, hsi: np.ndarray | None, chm: np.ndarray | None,
                 rng: np.random.Generator, augment: bool = False) -> dict[str, torch.Tensor]:
        """rgb (3,H,W) uint8 | hsi (B,h,w) reflectance | chm (H,W) metres; None = sensor missing."""
        s, d = self.s, self.dcfg
        rgb_shape, hsi_shape = tuple(s["rgb_shape"]), tuple(s["hsi_shape"])
        # ---- RGB
        has_rgb = rgb is not None and np.asarray(rgb).max() > 0
        if has_rgb:
            x = np.asarray(rgb, np.float32) / 255.0                                  # (3, H, W)
            if augment and (d.get("rgb_aug") or {}).get("enabled", False):
                a = d["rgb_aug"]
                x = np.clip(x * rng.lognormal(0, a.get("gain_sigma", 0.15))
                            * rng.lognormal(0, a.get("channel_sigma", 0.03), (3, 1, 1)), 0, 1)
            x = (x - self.rgb_mean) / self.rgb_std
        else:
            x = np.zeros(rgb_shape, np.float32)
        # ---- HSI
        has_hsi = hsi is not None and np.isfinite(np.asarray(hsi, np.float32)).any()
        if has_hsi:
            h = np.nan_to_num(np.asarray(hsi, np.float32), nan=0.0, posinf=0.0, neginf=0.0)  # (B, h, w)
            h = np.clip(h, 0.0, 1.5)
            if augment and (d.get("hsi_aug") or {}).get("enabled", False):
                h = spectral_augment(h, rng, d["hsi_aug"])
            if s.get("hsi_brightness_norm"):
                h = brightness_normalize(h, np.ones(h.shape[1:], bool))
            h = ((h - self.hsi_mean) / self.hsi_std).astype(np.float32)
        else:
            h = np.zeros(hsi_shape, np.float32)
        # ---- shared random 90° rotation of all sensors (train-time augmentation)
        r = int(rng.integers(4)) if augment else 0
        if r:
            x = np.ascontiguousarray(np.rot90(x, r, axes=(1, 2)))
            h = np.ascontiguousarray(np.rot90(h, r, axes=(1, 2)))
        # ---- LiDAR (height raster -> points)
        pts_out = np.zeros((self.n_points, 4), np.float32)
        mask = np.zeros(self.n_points, bool)
        k = 0
        if chm is not None:
            pts = chm_to_points(np.asarray(chm, np.float32).reshape(np.asarray(chm).shape[-2:]), self.px)
            k = len(pts)
            if k >= int(d["lidar_min_points"]):
                if r:                                                               # same rotation as images
                    a = r * np.pi / 2
                    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32)
                    pts[:, :2] = pts[:, :2] @ rot.T
                pts[:, :2] /= s["lidar_xy_std"]
                pts[:, 2] = (pts[:, 2] - s["lidar_z_mean"]) / s["lidar_z_std"]
                pts[:, 3] = (pts[:, 3] - s["lidar_i_mean"]) / s["lidar_i_std"]
                sel = rng.choice(k, self.n_points, replace=False) if k > self.n_points else np.arange(k)
                pts_out[: len(sel)] = pts[sel]
                mask[: len(sel)] = True
        has_lid = bool(mask.any())
        return {"rgb": torch.from_numpy(np.ascontiguousarray(x, np.float32)),
                "hsi": torch.from_numpy(np.ascontiguousarray(h, np.float32)),
                "lidar": torch.from_numpy(pts_out), "lidar_mask": torch.from_numpy(mask),
                "modality_mask": torch.tensor([has_rgb, has_hsi, has_lid], dtype=torch.float32),
                "hsi_valid_frac": torch.tensor(1.0 if has_hsi else 0.0), "lidar_count": torch.tensor(float(k))}


class DemoCrownDataset(Dataset):
    """Dataset over the demo_crowns layout with the same item contract as TreeCrownDataset."""

    def __init__(self, path: str | Path, indices: np.ndarray, stats: dict[str, Any], dcfg: dict[str, Any],
                 train: bool = False, seed: int = 0, meta: dict[str, Any] | None = None) -> None:
        self.path, self.indices = str(path), np.asarray(indices, np.int64)
        self.stats, self.dcfg, self.train, self.seed, self.epoch = stats, dcfg, train, int(seed), 0
        self.augment = train and bool(dcfg.get("augment", True))
        self.meta = meta or read_meta(path)
        self.labels_all, self.site_all = self.meta["labels"], self.meta["site"]
        self.num_classes = len(self.meta["names"])
        self.site_names = self.meta["site_names"]
        self.prep = CrownPreprocessor(stats, dcfg)

    @property
    def labels(self) -> np.ndarray:
        return self.labels_all[self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def raw(self, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Raw (rgb, hsi, chm) arrays of HDF5 row i."""
        a = load_arrays(self.path)
        return a["rgb"][i], a["hsi"][i].astype(np.float32), a["lidar"][i][0]

    def __getitem__(self, j: int) -> dict[str, torch.Tensor]:
        i = int(self.indices[j])
        rng = np.random.default_rng([self.seed, self.epoch, i]) if self.train else np.random.default_rng(i)
        item = self.prep(*self.raw(i), rng=rng, augment=self.augment)
        item.update({"label": torch.tensor(self.labels_all[i]), "height": torch.tensor(self.meta["height"][i]),
                     "crown_diameter": torch.tensor(self.meta["crown"][i]), "site": torch.tensor(self.site_all[i]),
                     "index": torch.tensor(i, dtype=torch.int64)})
        return item


def build_demo_datasets(cfg: dict[str, Any], refit_stats: bool = False
                        ) -> tuple[DemoCrownDataset, DemoCrownDataset, DemoCrownDataset, SiteSplit]:
    """Train/val/test datasets for the demo_crowns layout; stats fit on train and cached to JSON."""
    path, d = cfg["paths"]["data_file"], cfg["data"]
    meta = read_meta(path)
    split = build_demo_split(cfg, meta)
    sp_path = Path(cfg["paths"]["norm_stats"])
    stats = None
    if sp_path.exists() and not refit_stats:
        c = json.loads(sp_path.read_text(encoding="utf-8"))
        if (c.get("format") == FORMAT and c.get("version") == STATS_VERSION and c.get("data_file") == path
                and c.get("train_idx_hash") == int(split.train_idx.sum())
                and c.get("hsi_brightness_norm") == bool(d.get("hsi_brightness_norm", False))):
            stats = c
    if stats is None:
        stats = compute_demo_stats(path, split.train_idx, d, meta, seed=cfg["seed"])
        stats.update(data_file=path, train_idx_hash=int(split.train_idx.sum()), train_sites=split.train_sites)
        sp_path.parent.mkdir(parents=True, exist_ok=True)
        sp_path.write_text(json.dumps(stats), encoding="utf-8")
    rng = np.random.default_rng(cfg["seed"])

    def cap(ix: np.ndarray, n: int | None) -> np.ndarray:
        return ix if not n or n >= len(ix) else np.sort(rng.choice(ix, int(n), replace=False))

    mk = lambda ix, tr: DemoCrownDataset(path, ix, stats, d, tr, cfg["seed"], meta)  # noqa: E731
    return (mk(cap(split.train_idx, d.get("max_train_samples")), True),
            mk(cap(split.val_idx, d.get("max_eval_samples")), False),
            mk(cap(split.test_idx, d.get("max_eval_samples")), False), split)
