"""SYNTHETIC tree-crown dataset with REAL species names, in the `demo_tree_crowns.h5` layout.

*** The measurements are simulated. Species / site names are real; traits are plausible, not measured. ***

The output mirrors the attached reference file key-for-key, dtype-for-dtype and shape-for-shape:

    rgb             (N, 3, 32, 32)    uint8     reflectance × 255, 0.5 m pixels (16 m window)
    hsi             (N, 369, 8, 8)    float16   reflectance 0-1.2, 2 m pixels, 380-2510 nm
    lidar           (N, 1, 32, 32)    float32   canopy height model (m above ground), 0.5 m pixels
    species         (N,)              int64     index into attrs['species_names']
    site            (N,)              int64     index into attrs['site_names']
    height          (N,)              float32   tree height, m
    crown_diameter  (N,)              float32   m
    degraded        (N, 3)            bool      [rgb, hsi, lidar] sensor degraded for this crown
    wavelengths     (369,)            float64   nm
    root attrs: note, site_names, species_names  (+ extra descriptive attrs, see below)

Degradations (as in the reference file): RGB → strong colour noise; HSI → a block of bands
dropped to 0 plus extra noise; LiDAR → most returns missing (sparse raster).
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import h5py
import numpy as np

from src.data.species_catalog import SITES, SPECIES, Species
from src.data.synthetic import _smooth_noise

WINDOW_M = 16.0
RGB_PX, HSI_PX = 32, 8
P_DEGRADED = (0.20, 0.15, 0.09)           # rgb, hsi, lidar — matches the reference file


def _g(x: np.ndarray, mu: float, sig: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mu) / sig) ** 2)


def vegetation_spectrum(wl: np.ndarray, sp: Species, rng: np.random.Generator | None = None) -> np.ndarray:
    """Canopy reflectance (B,) from a species' shape parameters, with optional per-tree variation."""
    j = (lambda s: rng.normal(0, s)) if rng is not None else (lambda s: 0.0)
    green, edge, nir = sp.green * (1 + j(0.08)), sp.red_edge + j(2.5), sp.nir * (1 + j(0.07))
    swir, water = sp.swir * (1 + j(0.08)), np.clip(sp.water + j(0.05), 0, 1)
    vis = 0.03 + green * _g(wl, 550, 32) - 0.02 * _g(wl, 675, 18) + 0.01 * _g(wl, 450, 40)
    s = 1 / (1 + np.exp(-(wl - edge) / 11.0))                                  # red edge
    plateau = np.where(wl < 1300, nir, swir + (nir - swir) * np.exp(-(wl - 1300) / 350))
    absorb = 1 - water * (0.10 * _g(wl, 970, 25) + 0.18 * _g(wl, 1200, 35) + 0.75 * _g(wl, 1450, 45)
                          + 0.95 * _g(wl, 1940, 60) + 0.35 * _g(wl, 2500, 120))
    return np.clip((vis * (1 - s) + plateau * s) * absorb, 0.003, 0.9)


def soil_spectrum(wl: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Bare soil / litter: a gently rising ramp with weak water and clay features."""
    base = rng.uniform(0.10, 0.14) + rng.uniform(0.09, 0.13) * (wl - 380) / 2130
    return np.clip(base * (1 - 0.25 * _g(wl, 1940, 60) - 0.12 * _g(wl, 1450, 45) - 0.05 * _g(wl, 2200, 30)), 0.01, 1)


def _profile(rn: np.ndarray, shape: str) -> np.ndarray:
    """Normalized crown-surface height (1 at centre) vs normalized radius rn in [0, 1]."""
    rn = np.clip(rn, 0, 1)
    return {"dome": np.sqrt(1 - rn ** 2), "flat": (1 - rn ** 4) ** 0.5,
            "cone": 1 - rn, "column": 1 - rn ** 1.6}[shape]


def _cover(cx: float, cy: float, rad: float, n: int, sub: int = 4) -> np.ndarray:
    """Fraction of each of n×n pixels (16 m window) covered by a disk, by supersampling."""
    px = WINDOW_M / n
    c = (np.arange(n * sub) + 0.5) * px / sub - WINDOW_M / 2
    yy, xx = np.meshgrid(c, c, indexing="ij")
    inside = ((xx - cx) ** 2 + (yy - cy) ** 2) <= rad ** 2
    return inside.reshape(n, sub, n, sub).mean((1, 3))


def generate_real_species_dataset(path: str | Path, num_samples: int = 2000, seed: int = 42,
                                  species_codes: Sequence[str] | None = None,
                                  site_codes: Sequence[str] | None = None, min_per_species: int = 15) -> Path:
    """Write the dataset and return its path. ``species_codes`` / ``site_codes`` select subsets."""
    rng = np.random.default_rng(seed)
    species = [s for s in SPECIES if species_codes is None or s.code in species_codes]
    sites = [s for s in SITES if site_codes is None or s in site_codes]
    species = [s for s in species if any(t in sites for t in s.sites)]
    if not species:
        raise ValueError("no species occur at the selected sites")
    n, C, S = num_samples, len(species), len(sites)
    if n < C * min_per_species:
        raise ValueError(f"num_samples must be >= {C * min_per_species}")
    wl = np.linspace(380.0, 2510.0, 369)
    B = len(wl)
    ib = [int(np.argmin(np.abs(wl - w))) for w in (640, 550, 470)]            # RGB band centres

    # ---- who lives where: guarantee min_per_species, rest by site (uniform) x local abundance
    pairs: list[tuple[int, int]] = []
    for ci, sp in enumerate(species):
        homes = [sites.index(t) for t in sp.sites if t in sites]
        pairs += [(ci, int(rng.choice(homes))) for _ in range(min_per_species)]
    for _ in range(n - len(pairs)):
        si = int(rng.integers(S))
        local = [ci for ci, sp in enumerate(species) if sites[si] in sp.sites]
        w = np.array([species[ci].abundance for ci in local])
        pairs.append((int(rng.choice(local, p=w / w.sum())), si))
    order = rng.permutation(n)
    sp_idx = np.array([pairs[k][0] for k in order], np.int64)
    site_idx = np.array([pairs[k][1] for k in order], np.int64)

    site_gain = rng.uniform(0.88, 1.12, S)               # illumination / calibration per site
    site_tilt = rng.normal(0, 0.03, S)
    site_prod = rng.uniform(0.9, 1.1, S)                 # productivity -> height
    height = np.array([species[c].height * site_prod[s] for c, s in zip(sp_idx, site_idx)])
    height = np.clip(height * rng.lognormal(0, 0.18, n), 3, 60).astype(np.float32)
    crown = np.array([species[c].crown_ratio for c in sp_idx]) * height * rng.lognormal(0, 0.15, n)
    crown = np.clip(crown, 2.5, 14.0).astype(np.float32)
    degraded = rng.random((n, 3)) < np.array(P_DEGRADED)

    rgb = np.zeros((n, 3, RGB_PX, RGB_PX), np.uint8)
    hsi = np.zeros((n, B, HSI_PX, HSI_PX), np.float16)
    lid = np.zeros((n, 1, RGB_PX, RGB_PX), np.float32)
    c_r = (np.arange(RGB_PX) + 0.5) * WINDOW_M / RGB_PX - WINDOW_M / 2
    yy, xx = np.meshgrid(c_r, c_r, indexing="ij")
    for i in range(n):
        sp, si = species[sp_idx[i]], site_idx[i]
        rad, H = crown[i] / 2, float(height[i])
        cx, cy = rng.uniform(-1, 1, 2)
        tilt = 1 + site_tilt[si] * (wl - 1400) / 1100
        veg = vegetation_spectrum(wl, sp, rng) * site_gain[si] * tilt                  # (B,)
        soil = soil_spectrum(wl, rng) * site_gain[si] * tilt                           # (B,)
        # ---- hyperspectral (B, 8, 8): linear mixing of crown and soil by pixel cover
        cov = _cover(cx, cy, rad, HSI_PX)                                              # (8, 8)
        shade = 1 - 0.15 * np.clip((np.arange(HSI_PX)[:, None] - 3.5) / 4, -1, 1)     # sun from the north
        cube = cov[None] * veg[:, None, None] * shade[None] + (1 - cov[None]) * soil[:, None, None]
        cube = cube + rng.normal(0, 0.004, cube.shape)
        if degraded[i, 1]:
            b0 = int(rng.integers(0, B - 60))
            cube = cube + rng.normal(0, 0.03, cube.shape)
            cube[b0: b0 + int(rng.integers(30, 60))] = 0.0                              # dropped band block
        hsi[i] = np.clip(cube, 0, 1.2).astype(np.float16)
        # ---- RGB (3, 32, 32)
        rn = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / rad
        cov_r = _cover(cx, cy, rad, RGB_PX)
        tex = _smooth_noise(rng, RGB_PX, int(4 + 10 * sp.texture))
        crown_rgb = veg[ib][:, None, None] * np.array(sp.rgb_tint)[:, None, None]
        crown_rgb = crown_rgb * (1 + 0.35 * sp.texture * tex[None]) * (1.05 - 0.25 * np.clip(rn, 0, 1))[None]
        soil_rgb = soil[ib][:, None, None] * (1 + 0.25 * rng.normal(0, 1, (3, RGB_PX, RGB_PX)))
        img = cov_r[None] * crown_rgb + (1 - cov_r[None]) * soil_rgb
        img = img * 255 + rng.normal(0, 1.5, img.shape)
        if degraded[i, 0]:
            img = img + rng.normal(0, 25, img.shape)                                    # strong colour noise
        rgb[i] = np.clip(np.round(img), 0, 255).astype(np.uint8)
        # ---- LiDAR canopy height model (1, 32, 32)
        base = 0.25 if sp.shape == "cone" else 0.5
        z = H * (base + (1 - base) * _profile(rn, sp.shape))
        z = z * (1 + 0.06 * sp.texture * _smooth_noise(rng, RGB_PX, 8))
        chm = np.where(rn <= 1, z, 0.0) + rng.normal(0, 0.25, z.shape)                 # ground ≈ 0 ± noise
        if degraded[i, 2]:
            chm = np.where(rng.random(chm.shape) < 0.25, chm, 0.0)                     # sparse returns
        lid[i, 0] = chm.astype(np.float32)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    str_t = h5py.string_dtype()
    with h5py.File(path, "w") as f:
        f.attrs["note"] = ("SYNTHETIC demo data with real species and NEON site names; measurements are "
                           "simulated, not real NEON observations")
        f.attrs.create("site_names", np.array(sites, dtype=object), dtype=str_t)
        f.attrs.create("species_names", np.array([s.common for s in species], dtype=object), dtype=str_t)
        f.attrs.create("species_scientific_names", np.array([s.scientific for s in species], dtype=object), dtype=str_t)
        f.attrs.create("species_codes", np.array([s.code for s in species], dtype=object), dtype=str_t)
        f.attrs.create("species_families", np.array([s.family for s in species], dtype=object), dtype=str_t)
        f.attrs.create("species_leaf_type", np.array([s.leaf for s in species], dtype=object), dtype=str_t)
        f.attrs.create("site_full_names", np.array([SITES[s] for s in sites], dtype=object), dtype=str_t)
        f.attrs["pixel_size_m"] = np.array([WINDOW_M / RGB_PX, WINDOW_M / HSI_PX, WINDOW_M / RGB_PX])  # rgb, hsi, lidar
        f.attrs["synthetic"] = True
        f.attrs["generator_seed"] = seed
        kw = dict(compression="gzip")
        f.create_dataset("rgb", data=rgb, chunks=(min(250, n), 1, 8, 16), **kw)
        f.create_dataset("hsi", data=hsi, chunks=(min(125, n), 47, 2, 2), **kw)
        f.create_dataset("lidar", data=lid, chunks=(min(125, n), 1, 8, 8), **kw)
        f.create_dataset("species", data=sp_idx)
        f.create_dataset("site", data=site_idx)
        f.create_dataset("height", data=height)
        f.create_dataset("crown_diameter", data=crown)
        f.create_dataset("degraded", data=degraded)
        f.create_dataset("wavelengths", data=wl)
    return path
