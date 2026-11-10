"""Multimodal tree-crown dataset (RGB + HSI + LiDAR) with explicit invalid-value handling.

Invalid-value policy (applied before normalization):
  * RGB   — a crop that is all zeros is a missing sensor.
  * HSI   — values equal to ``hsi_nodata`` (-9999), non-finite, or > 1.5 reflectance are invalid;
            any invalid band marks that pixel invalid, invalid pixels are filled with the
            per-band train mean (i.e. 0 after z-scoring). If fewer than ``hsi_min_valid_frac``
            pixels are valid, the HSI sensor is treated as missing. Small negative reflectance is
            clipped to 0.
  * LiDAR — a point is valid iff x, y, z are finite. Non-finite intensity on a valid point is
            filled with the train intensity mean. Fewer than ``lidar_min_points`` valid points
            → LiDAR is missing. Points are subsampled / zero-padded to a fixed count with a mask.
A missing sensor is returned as an all-zero tensor with ``modality_mask[m] = 0``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.inspect import validate_schema
from src.data.splits import SiteSplit, site_held_out_split

MODALITIES: tuple[str, str, str] = ("rgb", "hsi", "lidar")
STATS_VERSION: int = 2  # bump when the cached norm-stats format changes


# ----------------------------------------------------------------------------- cleaning
def clean_hsi(raw: np.ndarray, scale: float, nodata: float) -> tuple[np.ndarray, np.ndarray]:
    """Convert raw HSI (B, h, w) to reflectance and a (h, w) valid-pixel mask. Invalid → NaN."""
    x = raw.astype(np.float32)
    bad = (x == nodata) | ~np.isfinite(x)
    x = x / scale
    bad |= x > 1.5
    x = np.where(bad, np.nan, np.clip(x, 0.0, None))
    valid_px = ~bad.any(axis=0)                       # (h, w)
    x[:, ~valid_px] = np.nan
    return x, valid_px


def brightness_normalize(x: np.ndarray, valid_px: np.ndarray) -> np.ndarray:
    """Divide every valid pixel spectrum (B, h, w) by its mean reflectance (removes multiplicative
    illumination / BRDF / calibration gain between flights and sites). Invalid pixels stay NaN."""
    m = np.ones((1,) + valid_px.shape, np.float32)                       # (1, h, w)
    m[0, valid_px] = x[:, valid_px].mean(axis=0)                          # mean over bands, valid px only
    return x / np.clip(m, 1e-3, None)                                    # (B, h, w); NaN px stay NaN


def spectral_augment(x: np.ndarray, rng: np.random.Generator, acfg: dict[str, float]) -> np.ndarray:
    """Random cross-site-style spectral shift on reflectance (B, h, w): global gain, linear spectral
    tilt across the band axis and a small additive offset. NaNs (invalid pixels) are preserved."""
    b = x.shape[0]
    gain = rng.lognormal(0.0, float(acfg.get("gain_sigma", 0.15)))
    tilt = rng.normal(0.0, float(acfg.get("tilt_sigma", 0.05)))
    ramp = np.linspace(-1.0, 1.0, b, dtype=np.float32)[:, None, None]   # (B, 1, 1)
    off = rng.normal(0.0, float(acfg.get("offset_sigma", 0.005)))
    return (x * gain * (1.0 + tilt * ramp) + off).astype(np.float32)    # (B, h, w)


def clean_lidar(raw: np.ndarray) -> np.ndarray:
    """Return only valid points (K, F) of a raw (P, F) cloud: rows with finite x, y, z."""
    ok = np.isfinite(raw[:, :3]).all(axis=1)
    return raw[ok]


# ------------------------------------------------------------------------ normalization
def compute_norm_stats(path: str | Path, indices: np.ndarray, dcfg: dict[str, Any], max_samples: int = 1000,
                       seed: int = 0) -> dict[str, list[float] | float]:
    """Fit per-modality normalization statistics on ``indices`` (the TRAIN split only)."""
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(indices, min(max_samples, len(indices)), replace=False))
    rgb_sum = np.zeros(3); rgb_sq = np.zeros(3); rgb_n = 0
    hsi_sum = None; hsi_sq = None; hsi_n = None
    lid: list[np.ndarray] = []
    with h5py.File(path, "r") as f:
        for i in idx:
            rgb = f["rgb"][i].astype(np.float64) / 255.0           # (3, H, W)
            if rgb.max() > 0:
                rgb_sum += rgb.sum((1, 2)); rgb_sq += (rgb ** 2).sum((1, 2)); rgb_n += rgb[0].size
            hsi, vpx = clean_hsi(f["hsi"][i], dcfg["hsi_scale"], dcfg["hsi_nodata"])
            if dcfg.get("hsi_brightness_norm", False):
                hsi = brightness_normalize(hsi, vpx)
            if vpx.any():
                v = hsi[:, vpx].astype(np.float64)                  # (B, n_valid)
                if hsi_sum is None:
                    hsi_sum = np.zeros(v.shape[0]); hsi_sq = np.zeros(v.shape[0]); hsi_n = 0
                hsi_sum += v.sum(1); hsi_sq += (v ** 2).sum(1); hsi_n += v.shape[1]
            pts = clean_lidar(f["lidar"][i])
            if len(pts):
                lid.append(pts)
    if rgb_n == 0 or hsi_n in (None, 0) or not lid:
        raise RuntimeError("not enough valid data in the train split to fit normalization stats")
    rgb_mean = rgb_sum / rgb_n
    hsi_mean = hsi_sum / hsi_n
    allp = np.concatenate(lid)
    inten = allp[:, 3][np.isfinite(allp[:, 3])]
    return {
        "rgb_mean": rgb_mean.tolist(),
        "rgb_std": np.sqrt(np.maximum(rgb_sq / rgb_n - rgb_mean ** 2, 1e-8)).tolist(),
        "hsi_mean": hsi_mean.tolist(),
        "hsi_std": np.sqrt(np.maximum(hsi_sq / hsi_n - hsi_mean ** 2, 1e-8)).tolist(),
        "lidar_xy_std": float(np.std(allp[:, :2]) + 1e-6),
        "lidar_z_mean": float(np.mean(allp[:, 2])),
        "lidar_z_std": float(np.std(allp[:, 2]) + 1e-6),
        "lidar_i_mean": float(np.mean(inten)),
        "lidar_i_std": float(np.std(inten) + 1e-6),
        "n_samples_used": int(len(idx)),
    }


# ------------------------------------------------------------------------------ dataset
class TreeCrownDataset(Dataset):
    """Reads crowns lazily from HDF5 and returns cleaned, normalized, fixed-shape tensors.

    Each item is a dict:
        rgb            (3, H, W)   float32
        hsi            (B, h, w)   float32
        lidar          (P, 4)      float32   x, y, z, intensity (normalized)
        lidar_mask     (P,)        bool      True = real point
        modality_mask  (3,)        float32   [rgb, hsi, lidar] 1 = present
        hsi_valid_frac ()          float32
        lidar_count    ()          float32   valid points before subsampling
        label          ()          int64
        height         ()          float32   m (raw, for the auxiliary head)
        crown_diameter ()          float32   m
        site           ()          int64     index into ``self.site_names``
        index          ()          int64     row in the HDF5 file
    """

    def __init__(self, path: str | Path, indices: np.ndarray, stats: dict[str, Any], dcfg: dict[str, Any],
                 train: bool = False, seed: int = 0) -> None:
        self.path = str(path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.dcfg = dcfg
        self.train = train
        self.seed = int(seed)
        self.epoch = 0  # set by the trainer each epoch (augmentation RNG)
        self.augment = train and bool(dcfg.get("augment", True))
        self.n_points = int(dcfg["lidar_num_points"])
        self._h5: h5py.File | None = None
        with h5py.File(self.path, "r") as f:
            self.labels_all = f["labels/species_id"][:].astype(np.int64)
            self.height_all = f["labels/height"][:].astype(np.float32)
            self.crown_all = f["labels/crown_diameter"][:].astype(np.float32)
            sites = np.array([s.decode() for s in f["meta/site"][:]])
            self.num_classes = (len(f["species_names"]) if "species_names" in f
                                else int(self.labels_all.max()) + 1)
        self.site_names = sorted(set(sites.tolist()))
        self.site_all = np.array([self.site_names.index(s) for s in sites], dtype=np.int64)
        self.rgb_mean = np.asarray(stats["rgb_mean"], np.float32)[:, None, None]
        self.rgb_std = np.asarray(stats["rgb_std"], np.float32)[:, None, None]
        self.hsi_mean = np.asarray(stats["hsi_mean"], np.float32)[:, None, None]
        self.hsi_std = np.asarray(stats["hsi_std"], np.float32)[:, None, None]
        self.stats = stats

    @property
    def labels(self) -> np.ndarray:
        """Species ids of the samples in this split, (len(self),)."""
        return self.labels_all[self.indices]

    def _file(self) -> h5py.File:
        if self._h5 is None:  # opened lazily so each DataLoader worker gets its own handle
            self._h5 = h5py.File(self.path, "r")
        return self._h5

    def __len__(self) -> int:
        return len(self.indices)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def _load_rgb(self, i: int, rng: np.random.Generator) -> tuple[np.ndarray, bool]:
        raw = self._file()["rgb"][i]                                       # (3, H, W) uint8
        if raw.max() == 0:
            return np.zeros(raw.shape, np.float32), False
        x = raw.astype(np.float32) / 255.0                                 # (3, H, W) in [0, 1]
        aug = self.dcfg.get("rgb_aug") or {}
        if self.augment and aug.get("enabled", False):                     # brightness / colour jitter
            g = rng.lognormal(0.0, float(aug.get("gain_sigma", 0.15)))
            cg = rng.lognormal(0.0, float(aug.get("channel_sigma", 0.03)), (3, 1, 1))
            x = np.clip(x * g * cg, 0.0, 1.0).astype(np.float32)           # (3, H, W)
        return (x - self.rgb_mean) / self.rgb_std, True                    # (3, H, W)

    def _load_hsi(self, i: int, rng: np.random.Generator) -> tuple[np.ndarray, bool, float]:
        x, vpx = clean_hsi(self._file()["hsi"][i], self.dcfg["hsi_scale"], self.dcfg["hsi_nodata"])
        frac = float(vpx.mean())                                           # (B, h, w), (h, w)
        if frac < float(self.dcfg["hsi_min_valid_frac"]):
            return np.zeros(x.shape, np.float32), False, frac
        aug = self.dcfg.get("hsi_aug") or {}
        if self.augment and aug.get("enabled", False):
            x = spectral_augment(x, rng, aug)                              # (B, h, w)
        if self.dcfg.get("hsi_brightness_norm", False):
            x = brightness_normalize(x, vpx)                               # (B, h, w)
        x = (x - self.hsi_mean) / self.hsi_std                             # (B, h, w)
        return np.nan_to_num(x, nan=0.0).astype(np.float32), True, frac    # invalid px -> band mean

    def _load_lidar(self, i: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, bool, int]:
        pts = clean_lidar(self._file()["lidar"][i])                        # (K, 4)
        k = len(pts)
        out = np.zeros((self.n_points, 4), np.float32)
        mask = np.zeros(self.n_points, bool)
        if k < int(self.dcfg["lidar_min_points"]):
            return out, mask, False, k
        s = self.stats
        pts = pts.copy()
        pts[~np.isfinite(pts[:, 3]), 3] = s["lidar_i_mean"]                # fill bad intensity
        pts[:, :2] -= pts[:, :2].mean(0, keepdims=True)                    # recentre crown in xy
        pts[:, :2] /= s["lidar_xy_std"]
        pts[:, 2] = (pts[:, 2] - s["lidar_z_mean"]) / s["lidar_z_std"]     # keep absolute height info
        pts[:, 3] = (pts[:, 3] - s["lidar_i_mean"]) / s["lidar_i_std"]
        sel = rng.choice(k, self.n_points, replace=False) if k > self.n_points else np.arange(k)
        out[: len(sel)] = pts[sel]                                         # (P, 4)
        mask[: len(sel)] = True                                            # (P,)
        return out, mask, True, k

    def __getitem__(self, j: int) -> dict[str, torch.Tensor]:
        i = int(self.indices[j])
        # Seeded per (seed, epoch, row): training augmentation varies by epoch yet is reproducible;
        # evaluation items are fixed.
        rng = (np.random.default_rng([self.seed, self.epoch, i]) if self.train
               else np.random.default_rng(i))
        rgb, has_rgb = self._load_rgb(i, rng)
        hsi, has_hsi, hsi_frac = self._load_hsi(i, rng)
        lid, lmask, has_lid, lcount = self._load_lidar(i, rng)
        if self.augment:
            kr = int(rng.integers(4))
            flip = bool(rng.integers(2))
            rgb = np.rot90(rgb, kr, axes=(1, 2))                           # (3, H, W)
            hsi = np.rot90(hsi, kr, axes=(1, 2))                           # (B, h, w)
            ang = kr * np.pi / 2
            rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], np.float32)
            lid[:, :2] = lid[:, :2] @ rot.T                                # (P, 2)
            if flip:
                rgb, hsi = rgb[:, :, ::-1], hsi[:, :, ::-1]
                lid[:, 0] = -lid[:, 0]
            lid[~lmask] = 0.0
        return {
            "rgb": torch.from_numpy(np.ascontiguousarray(rgb, dtype=np.float32)),
            "hsi": torch.from_numpy(np.ascontiguousarray(hsi, dtype=np.float32)),
            "lidar": torch.from_numpy(lid),
            "lidar_mask": torch.from_numpy(lmask),
            "modality_mask": torch.tensor([has_rgb, has_hsi, has_lid], dtype=torch.float32),
            "hsi_valid_frac": torch.tensor(hsi_frac, dtype=torch.float32),
            "lidar_count": torch.tensor(float(lcount)),
            "label": torch.tensor(self.labels_all[i]),
            "height": torch.tensor(self.height_all[i]),
            "crown_diameter": torch.tensor(self.crown_all[i]),
            "site": torch.tensor(self.site_all[i]),
            "index": torch.tensor(i, dtype=torch.int64),
        }


# --------------------------------------------------------------------------- builders
def is_demo_format(cfg: dict[str, Any]) -> bool:
    """True when the config points at the compact demo_crowns layout (see src/data/crown_dataset.py)."""
    return cfg.get("data", {}).get("format", "neon_mmtc") == "demo_crowns"


def build_split(cfg: dict[str, Any]) -> SiteSplit:
    """Read sites/labels from the data file and build the site-held-out split from config."""
    if is_demo_format(cfg):
        from src.data.crown_dataset import build_demo_split
        return build_demo_split(cfg)
    path = cfg["paths"]["data_file"]
    validate_schema(path)
    with h5py.File(path, "r") as f:
        sites = f["meta/site"][:]
        labels = f["labels/species_id"][:]
    sc = cfg["split"]
    return site_held_out_split(sites, labels, sc.get("test_sites"), sc.get("val_sites"),
                               sc["test_frac"], sc["val_frac"],
                               int(sc["seed"]) if sc.get("seed") is not None else cfg["seed"])


def build_datasets(cfg: dict[str, Any], refit_stats: bool = False
                   ) -> tuple[TreeCrownDataset, TreeCrownDataset, TreeCrownDataset, SiteSplit]:
    """Create train/val/test datasets. Normalization stats are fit on train and cached to JSON."""
    if is_demo_format(cfg):
        from src.data.crown_dataset import build_demo_datasets
        return build_demo_datasets(cfg, refit_stats)
    split = build_split(cfg)
    stats_path = Path(cfg["paths"]["norm_stats"])
    stats: dict[str, Any] | None = None
    if stats_path.exists() and not refit_stats:
        cached = json.loads(stats_path.read_text(encoding="utf-8"))
        if (cached.get("train_sites") == split.train_sites and cached.get("data_file") == cfg["paths"]["data_file"]
                and cached.get("version") == STATS_VERSION
                and cached.get("hsi_brightness_norm") == bool(cfg["data"].get("hsi_brightness_norm", False))):
            stats = cached
    if stats is None:
        stats = compute_norm_stats(cfg["paths"]["data_file"], split.train_idx, cfg["data"], seed=cfg["seed"])
        with h5py.File(cfg["paths"]["data_file"], "r") as f:
            h = f["labels/height"][:][split.train_idx]
            cd = f["labels/crown_diameter"][:][split.train_idx]
        stats.update(height_mean=float(np.mean(h)), height_std=float(np.std(h) + 1e-6),
                     crown_mean=float(np.mean(cd)), crown_std=float(np.std(cd) + 1e-6),
                     train_sites=split.train_sites, data_file=cfg["paths"]["data_file"], version=STATS_VERSION,
                     hsi_brightness_norm=bool(cfg["data"].get("hsi_brightness_norm", False)))
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats), encoding="utf-8")
    path, d = cfg["paths"]["data_file"], cfg["data"]
    rng = np.random.default_rng(cfg["seed"])

    def cap(idx: np.ndarray, n: int | None) -> np.ndarray:
        """Optional seeded subset (smoke tests / quick runs); None = all."""
        return idx if not n or n >= len(idx) else np.sort(rng.choice(idx, int(n), replace=False))

    return (TreeCrownDataset(path, cap(split.train_idx, d.get("max_train_samples")), stats, d, True, cfg["seed"]),
            TreeCrownDataset(path, cap(split.val_idx, d.get("max_eval_samples")), stats, d, False, cfg["seed"]),
            TreeCrownDataset(path, cap(split.test_idx, d.get("max_eval_samples")), stats, d, False, cfg["seed"]),
            split)


def build_dataloaders(cfg: dict[str, Any], sampler: torch.utils.data.Sampler | None = None
                      ) -> tuple[DataLoader, DataLoader, DataLoader, SiteSplit]:
    """DataLoaders for train/val/test. ``sampler`` (e.g. class-balanced, Phase 4) replaces shuffling."""
    tr, va, te, split = build_datasets(cfg)
    d = cfg["data"]
    g = torch.Generator().manual_seed(cfg["seed"])
    kw = dict(batch_size=d["batch_size"], num_workers=d["num_workers"], pin_memory=torch.cuda.is_available())
    return (DataLoader(tr, sampler=sampler, shuffle=sampler is None, drop_last=False, generator=g, **kw),
            DataLoader(va, shuffle=False, **kw), DataLoader(te, shuffle=False, **kw), split)


def make_eval_dataset(cfg: dict[str, Any], stats: dict[str, Any], split_name: str
                      ) -> tuple[Dataset, SiteSplit, dict[str, Any]]:
    """Dataset for 'val'/'test' with a checkpoint's stats, plus layout-independent label info:
    names, all labels, degraded flags (or None), synthetic flag, train indices."""
    split = build_split(cfg)
    idx = {"val": split.val_idx, "test": split.test_idx}[split_name]
    path = cfg["paths"]["data_file"]
    if is_demo_format(cfg):
        from src.data.crown_dataset import DemoCrownDataset, read_meta
        meta = read_meta(path)
        ds = DemoCrownDataset(path, idx, stats, cfg["data"], train=False, meta=meta)
        info = {"names": meta["names"], "labels": meta["labels"], "degraded": meta["degraded"],
                "synthetic": meta["synthetic"]}
    else:
        ds = TreeCrownDataset(path, idx, stats, cfg["data"], train=False)
        with h5py.File(path, "r") as f:
            info = {"names": [s.decode() for s in f["species_names"][:]] if "species_names" in f else None,
                    "labels": f["labels/species_id"][:],
                    "degraded": f["synthetic_truth/degraded"][:] if "synthetic_truth/degraded" in f else None,
                    "synthetic": bool(f.attrs.get("synthetic", False))}
    return ds, split, info
