"""Inspect an HDF5 file (groups, keys, shapes, dtypes) and validate the expected schema."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

# Keys the dataset class reads. Shapes are *read from the file*, never assumed;
# only their rank/consistency is checked here.
REQUIRED_KEYS: dict[str, int] = {
    "rgb": 4,                      # (N, 3, H, W)
    "hsi": 4,                      # (N, B, h, w)
    "lidar": 3,                    # (N, P, F)
    "lidar_num_points": 1,         # (N,)
    "labels/species_id": 1,        # (N,)
    "labels/height": 1,            # (N,)
    "labels/crown_diameter": 1,    # (N,)
    "meta/site": 1,                # (N,)
}


def describe_h5(path: str | Path) -> list[dict[str, Any]]:
    """Return one record per group/dataset: name, kind, shape, dtype, attrs."""
    out: list[dict[str, Any]] = []

    def visit(name: str, obj: h5py.HLObject) -> None:
        rec: dict[str, Any] = {"name": name, "attrs": dict(obj.attrs)}
        if isinstance(obj, h5py.Dataset):
            rec.update(kind="dataset", shape=obj.shape, dtype=str(obj.dtype))
        else:
            rec.update(kind="group", shape=None, dtype=None)
        out.append(rec)

    with h5py.File(path, "r") as f:
        out.append({"name": "/", "kind": "group", "shape": None, "dtype": None, "attrs": dict(f.attrs)})
        f.visititems(visit)
    return out


def print_h5_summary(path: str | Path, sample_stats: bool = True) -> None:
    """Pretty-print the file tree and, optionally, value ranges / NaN counts of small samples."""
    print(f"HDF5 file: {path}")
    for rec in describe_h5(path):
        if rec["kind"] == "group":
            print(f"[group]   {rec['name']}")
        else:
            print(f"[dataset] {rec['name']:<28} shape={str(rec['shape']):<24} dtype={rec['dtype']}")
        for k, v in rec["attrs"].items():
            print(f"            @{k} = {v}")
    if not sample_stats:
        return
    print("\nValue checks on first 64 rows:")
    with h5py.File(path, "r") as f:
        for key in ("rgb", "hsi", "lidar"):
            if key not in f:
                continue
            a = f[key][:64].astype(np.float64)
            print(f"  {key:<6} min={np.nanmin(a):.3f} max={np.nanmax(a):.3f} "
                  f"NaN={int(np.isnan(a).sum())} -9999={int((a == -9999).sum())}")
        if "labels/species_id" in f:
            y = f["labels/species_id"][:]
            print(f"  species: {len(np.unique(y))} unique ids in [{y.min()}, {y.max()}]")
        if "meta/site" in f:
            s, cnt = np.unique(f["meta/site"][:], return_counts=True)
            print("  sites: " + ", ".join(f"{a.decode()}={b}" for a, b in zip(s, cnt)))


def validate_schema(path: str | Path) -> dict[str, tuple[int, ...]]:
    """Check required keys exist with the right rank and a common N. Return their shapes."""
    shapes: dict[str, tuple[int, ...]] = {}
    with h5py.File(path, "r") as f:
        for key, rank in REQUIRED_KEYS.items():
            if key not in f:
                raise KeyError(f"required key '{key}' missing from {path}")
            shp = f[key].shape
            if len(shp) != rank:
                raise ValueError(f"'{key}' has rank {len(shp)}, expected {rank} (shape {shp})")
            shapes[key] = shp
    ns = {k: v[0] for k, v in shapes.items()}
    if len(set(ns.values())) != 1:
        raise ValueError(f"inconsistent sample counts: {ns}")
    return shapes
