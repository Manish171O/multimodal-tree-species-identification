"""SYNTHETIC demo-data generator mimicking the NEON Multi-Modal Tree Crown layout.

*** This data is SYNTHETIC. It is NOT real NEON data. ***

It exists so the full pipeline (loading -> training -> evaluation) is testable end
to end without the real dataset. Signals are physically *inspired* (vegetation
spectra with a red edge, species allometry, crown-shape LiDAR point clouds, site
illumination shifts, long-tailed species frequencies) but numbers produced on this
data say nothing about real-world performance.

HDF5 layout written (all arrays indexed by crown ``i`` along axis 0):

    /rgb                     (N, 3, S_rgb, S_rgb)   uint8    0 everywhere = missing
    /hsi                     (N, B, S_hsi, S_hsi)   int16    reflectance*10000, -9999 = nodata
    /hsi_wavelengths         (B,)                   float32  nm
    /lidar                   (N, P, 4)              float32  x, y [m, crown-centred], z [m AGL],
                                                             intensity; NaN rows = padding
    /lidar_num_points        (N,)                   int32
    /labels/species_id       (N,)                   int64    0..C-1
    /labels/height           (N,)                   float32  m
    /labels/crown_diameter   (N,)                   float32  m
    /meta/site               (N,)                   bytes    NEON-style 4-letter site code
    /meta/plot_id            (N,)                   bytes
    /meta/individual_id      (N,)                   bytes
    /species_names           (C,)                   bytes
    /site_names              (n_sites,)             bytes
    /synthetic_truth/missing   (N, 3) bool  [rgb, hsi, lidar] fully missing
    /synthetic_truth/degraded  (N, 3) bool  [rgb, hsi, lidar] degraded quality
    root attrs: synthetic=True, description, generator_seed
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

SITE_CODES: list[str] = [
    "HARV", "BART", "OSBS", "TALL", "MLBS", "DELA", "SERC", "SCBI",
    "UNDE", "STEI", "TREE", "BONA", "JERC", "LENO", "NIWO", "WREF",
    "ABBY", "SJER", "SOAP", "TEAK", "RMNP", "YELL", "GRSM", "CLBJ",
]


@dataclass
class SyntheticConfig:
    """Parameters for the synthetic generator (mirrors ``synthetic:`` in the YAML)."""

    num_samples: int = 2000
    num_species: int = 167
    num_sites: int = 16
    rgb_size: int = 64
    hsi_size: int = 12
    hsi_bands: int = 369
    lidar_points: int = 1024
    long_tail_alpha: float = 1.1
    p_missing_rgb: float = 0.03
    p_missing_hsi: float = 0.05
    p_missing_lidar: float = 0.05
    p_degraded: float = 0.20
    p_nodata_pixels: float = 0.02


def neon_like_wavelengths(n_bands: int = 369) -> np.ndarray:
    """NEON NIS-like band centres (380-2510 nm, 5 nm) minus water-absorption bands."""
    wl = np.linspace(380.0, 2510.0, 426)
    keep = ~(((wl > 1340) & (wl < 1445)) | ((wl > 1790) & (wl < 1955)))
    wl = wl[keep]
    # Trim noisy detector edges symmetrically until we hit exactly n_bands.
    excess = len(wl) - n_bands
    if excess < 0:
        raise ValueError(f"cannot produce {n_bands} bands from NEON-like grid ({len(wl)})")
    lo = excess // 2
    return wl[lo: lo + n_bands].astype(np.float32)


def _gauss(x: np.ndarray, mu: float, sig: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mu) / sig) ** 2)


def _vegetation_spectrum(wl: np.ndarray, rng: np.random.Generator, parent: dict | None) -> dict:
    """Sample parameters of a smooth leaf/canopy reflectance curve (genus or species)."""
    jitter = (lambda v, s: v + rng.normal(0, s)) if parent else (lambda v, s: v)
    p = parent or {
        "green": rng.uniform(0.06, 0.14), "red_edge": rng.uniform(695, 730),
        "edge_w": rng.uniform(9, 16), "nir": rng.uniform(0.28, 0.55),
        "swir": rng.uniform(0.15, 0.32), "water": rng.uniform(0.3, 0.7),
        "bumps": [(rng.uniform(450, 2400), rng.uniform(30, 150), rng.normal(0, 0.02)) for _ in range(4)],
    }
    q = {
        "green": jitter(p["green"], 0.012), "red_edge": jitter(p["red_edge"], 4.0),
        "edge_w": jitter(p["edge_w"], 1.0), "nir": jitter(p["nir"], 0.03),
        "swir": jitter(p["swir"], 0.02), "water": jitter(p["water"], 0.05),
        "bumps": [(m + (rng.normal(0, 20) if parent else 0), s, a + (rng.normal(0, 0.012) if parent else 0))
                  for m, s, a in p["bumps"]],
    }
    return q


def _render_spectrum(wl: np.ndarray, q: dict) -> np.ndarray:
    """Turn spectrum parameters into reflectance over ``wl`` (shape (B,))."""
    vis = 0.04 + q["green"] * _gauss(wl, 550, 35) - 0.025 * _gauss(wl, 670, 20)
    edge = 1.0 / (1.0 + np.exp(-(wl - q["red_edge"]) / q["edge_w"]))
    nir = q["nir"] * edge * np.where(wl < 1300, 1.0, 0.0)
    swir_level = q["swir"] + (q["nir"] - q["swir"]) * np.exp(-(wl - 1300) / 600)
    swir = np.where(wl >= 1300, swir_level, 0.0) * edge
    water = 1 - q["water"] * (0.25 * _gauss(wl, 1200, 40) + 0.35 * _gauss(wl, 1700, 60) + 0.5 * _gauss(wl, 2150, 120))
    r = (vis * (1 - edge) + nir + swir) * water
    for m, s, a in q["bumps"]:
        r = r + a * _gauss(wl, m, s)
    return np.clip(r, 0.005, 0.9)


def _smooth_noise(rng: np.random.Generator, size: int, cells: int) -> np.ndarray:
    """Cheap smooth 2-D noise: coarse grid upsampled bilinearly. Returns (size, size)."""
    cells = max(2, cells)
    g = rng.normal(0, 1, (cells + 1, cells + 1))
    t = np.linspace(0, cells, size, endpoint=False)
    i0 = t.astype(int)
    f = t - i0
    a = g[i0][:, i0] * (1 - f)[None, :] + g[i0][:, i0 + 1] * f[None, :]
    b = g[i0 + 1][:, i0] * (1 - f)[None, :] + g[i0 + 1][:, i0 + 1] * f[None, :]
    return a * (1 - f)[:, None] + b * f[:, None]


def _box_blur(img: np.ndarray, k: int) -> np.ndarray:
    """Separable box blur of a (C, H, W) array with window k (edge padded)."""
    pad = k // 2
    out = np.pad(img, ((0, 0), (pad, pad), (pad, pad)), mode="edge").astype(np.float32)
    ker = np.ones(k, dtype=np.float32) / k
    out = np.apply_along_axis(lambda v: np.convolve(v, ker, mode="valid"), 2, out)
    out = np.apply_along_axis(lambda v: np.convolve(v, ker, mode="valid"), 1, out)
    return out


def generate_synthetic_dataset(path: str | Path, cfg: SyntheticConfig, seed: int = 42) -> Path:
    """Generate the synthetic HDF5 file at ``path`` and return the path."""
    rng = np.random.default_rng(seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n, c = cfg.num_samples, cfg.num_species
    if cfg.num_sites > len(SITE_CODES):
        raise ValueError(f"num_sites <= {len(SITE_CODES)}")
    sites = SITE_CODES[: cfg.num_sites]
    wl = neon_like_wavelengths(cfg.hsi_bands)
    B, S, R, P = cfg.hsi_bands, cfg.hsi_size, cfg.rgb_size, cfg.lidar_points

    # --- species definitions: genus-level clusters -> realistic confusions -------------
    n_genera = max(1, c // 6)
    genus_of = rng.integers(0, n_genera, c)
    genus_spec = [_vegetation_spectrum(wl, rng, None) for _ in range(n_genera)]
    genus_conifer = rng.random(n_genera) < 0.35
    sp_spec = np.stack([_render_spectrum(wl, _vegetation_spectrum(wl, rng, genus_spec[g])) for g in genus_of])
    conifer = genus_conifer[genus_of]
    sp_height = np.where(conifer, rng.uniform(12, 40, c), rng.uniform(6, 32, c))
    sp_allom = np.where(conifer, rng.uniform(0.18, 0.28, c), rng.uniform(0.30, 0.50, c))  # crown_d / height
    sp_shape = np.where(conifer, rng.uniform(0.8, 1.3, c), rng.uniform(1.8, 3.5, c))       # crown profile exp
    sp_depth = np.where(conifer, rng.uniform(0.55, 0.8, c), rng.uniform(0.3, 0.55, c))      # crown depth / H
    sp_intensity = rng.uniform(0.2, 0.8, c)
    sp_texture = rng.integers(3, 14, c)
    sp_rgb_tint = rng.normal(0, 0.08, (c, 3))

    # --- biogeography: each species lives in 2-5 sites --------------------------------
    sp_sites = [rng.choice(cfg.num_sites, size=int(rng.integers(2, 6)), replace=False) for _ in range(c)]
    site_gain = rng.uniform(0.8, 1.2, cfg.num_sites)          # illumination / BRDF
    site_tilt = rng.normal(0, 0.03, cfg.num_sites)             # spectral calibration tilt
    site_height = rng.uniform(0.85, 1.15, cfg.num_sites)       # productivity
    bg_spec = _render_spectrum(wl, _vegetation_spectrum(wl, rng, None)) * 0.5 + 0.05  # understory/soil

    # --- long-tailed species frequencies, at least 2 crowns per species ---------------
    freq = 1.0 / np.arange(1, c + 1) ** cfg.long_tail_alpha
    freq = freq[rng.permutation(c)]
    freq /= freq.sum()
    if n < 2 * c:
        raise ValueError("num_samples must be >= 2 * num_species")
    species = np.concatenate([np.repeat(np.arange(c), 2), rng.choice(c, n - 2 * c, p=freq)])
    species = species[rng.permutation(n)].astype(np.int64)
    site_idx = np.array([rng.choice(sp_sites[s]) for s in species])

    height = (sp_height[species] * site_height[site_idx] * rng.lognormal(0, 0.15, n)).astype(np.float32)
    crown_d = (height * sp_allom[species] * rng.lognormal(0, 0.15, n)).clip(1.5, 14).astype(np.float32)

    missing = np.stack([rng.random(n) < cfg.p_missing_rgb,
                        rng.random(n) < cfg.p_missing_hsi,
                        rng.random(n) < cfg.p_missing_lidar], 1)
    missing[missing.all(1), 0] = False  # never drop all three sensors
    degraded = np.zeros((n, 3), bool)
    deg_rows = np.where(rng.random(n) < cfg.p_degraded)[0]
    degraded[deg_rows, rng.integers(0, 3, len(deg_rows))] = True
    degraded &= ~missing

    yy_h, xx_h = np.mgrid[0:S, 0:S] - (S - 1) / 2.0           # HSI grid, 1 m pixels
    yy_r, xx_r = (np.mgrid[0:R, 0:R] - (R - 1) / 2.0) * 0.1   # RGB grid, 0.1 m pixels
    band_rgb = [int(np.argmin(np.abs(wl - w))) for w in (650, 550, 470)]

    with h5py.File(path, "w") as f:
        f.attrs["synthetic"] = True
        f.attrs["description"] = ("SYNTHETIC demo data mimicking NEON Multi-Modal Tree Crown layout. "
                                  "Not real observations.")
        f.attrs["generator_seed"] = seed
        d_rgb = f.create_dataset("rgb", (n, 3, R, R), np.uint8, chunks=(1, 3, R, R), compression="gzip")
        d_hsi = f.create_dataset("hsi", (n, B, S, S), np.int16, chunks=(1, B, S, S), compression="gzip")
        d_lid = f.create_dataset("lidar", (n, P, 4), np.float32, chunks=(1, P, 4), compression="gzip")
        npts_all = np.zeros(n, np.int32)

        for i in range(n):
            s, st = species[i], site_idx[i]
            rad = crown_d[i] / 2.0
            # ---------------- HSI ----------------
            spec = sp_spec[s] * rng.lognormal(0, 0.06) * (1 + rng.normal(0, 0.02, B))
            spec = spec * site_gain[st] * (1 + site_tilt[st] * (wl - 1400) / 1100)
            dist = np.sqrt(yy_h ** 2 + xx_h ** 2)
            cover = np.clip(rad + 0.5 - dist, 0, 1)                          # (S,S) soft crown mask
            shade = 1 - 0.25 * np.clip((yy_h + xx_h) / (S + 1e-6), -1, 1)     # sun-side gradient
            cube = (cover[None] * spec[:, None, None] * shade[None]
                    + (1 - cover[None]) * bg_spec[:, None, None] * site_gain[st])
            cube = cube + rng.normal(0, 0.004, cube.shape)
            if degraded[i, 1]:  # thin cloud / haze: flatten + heavy noise
                cube = 0.5 * cube + 0.5 * cube.mean(0, keepdims=True) + rng.normal(0, 0.03, cube.shape)
            hsi = np.round(np.clip(cube, 0, 1.2) * 10000).astype(np.int16)
            nodata = rng.random((S, S)) < cfg.p_nodata_pixels
            hsi[:, nodata] = -9999
            if missing[i, 1]:
                hsi[:] = -9999
            d_hsi[i] = hsi

            # ---------------- RGB ----------------
            rgb_base = spec[band_rgb] / spec[band_rgb].max() * 0.6 + sp_rgb_tint[s]
            tex = _smooth_noise(rng, R, int(sp_texture[s]))
            dist_r = np.sqrt(yy_r ** 2 + xx_r ** 2)
            cov_r = np.clip((rad - dist_r) / 0.3 + 0.5, 0, 1)
            img = cov_r[None] * (rgb_base[:, None, None] * (1 + 0.25 * tex[None]))
            img = img + (1 - cov_r[None]) * np.array([0.35, 0.33, 0.25])[:, None, None]
            img = img * site_gain[st] * (1 - 0.3 * np.clip((yy_r + xx_r) / 6.4, -1, 1))[None]
            img = img + rng.normal(0, 0.02, img.shape)
            if degraded[i, 0]:  # motion blur + over-exposure
                img = np.clip(_box_blur(img, 7) * rng.uniform(1.4, 1.8), 0, 1)
            rgb = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            if missing[i, 0]:
                rgb[:] = 0
            d_rgb[i] = rgb

            # ---------------- LiDAR --------------
            density = 8.0 if not degraded[i, 2] else 0.5                # pts / m^2
            k = int(np.clip(rng.poisson(density * np.pi * rad ** 2) + 20, 0, P))
            if degraded[i, 2]:
                k = int(rng.integers(5, 40))
            if missing[i, 2]:
                k = 0
            pts = np.full((P, 4), np.nan, np.float32)
            if k > 0:
                r_ = rad * np.sqrt(rng.random(k))
                th = rng.uniform(0, 2 * np.pi, k)
                depth = sp_depth[s] * height[i]
                z = height[i] - depth * (r_ / rad) ** sp_shape[s]
                z = z - rng.exponential(0.15 * depth, k) * (rng.random(k) < 0.3)  # interior returns
                ground = rng.random(k) < 0.08
                z[ground] = rng.uniform(0, 0.3, ground.sum())
                noise = 0.15 if not degraded[i, 2] else 0.8
                inten = np.clip(sp_intensity[s] + rng.normal(0, 0.08, k), 0, 1)
                pts[:k] = np.stack([r_ * np.cos(th) + rng.normal(0, noise, k),
                                    r_ * np.sin(th) + rng.normal(0, noise, k),
                                    np.clip(z + rng.normal(0, noise, k), 0, None), inten], 1)
                bad = rng.random(k) < 0.005                                # sporadic invalid returns
                pts[:k][bad, 3] = np.nan
            d_lid[i] = pts
            npts_all[i] = k

        f.create_dataset("hsi_wavelengths", data=wl)
        f.create_dataset("lidar_num_points", data=npts_all)
        g = f.create_group("labels")
        g.create_dataset("species_id", data=species)
        g.create_dataset("height", data=height)
        g.create_dataset("crown_diameter", data=crown_d)
        m = f.create_group("meta")
        m.create_dataset("site", data=np.array([sites[j] for j in site_idx], dtype="S4"))
        m.create_dataset("plot_id", data=np.array([f"{sites[j]}_{rng.integers(1, 40):03d}" for j in site_idx], dtype="S8"))
        m.create_dataset("individual_id", data=np.array([f"SYN.{i:06d}" for i in range(n)], dtype="S10"))
        f.create_dataset("species_names", data=np.array([f"SYN_SP{j:03d}_G{genus_of[j]:02d}" for j in range(c)], dtype="S16"))
        f.create_dataset("site_names", data=np.array(sites, dtype="S4"))
        t = f.create_group("synthetic_truth")
        t.create_dataset("missing", data=missing)
        t.create_dataset("degraded", data=degraded)
    return path
