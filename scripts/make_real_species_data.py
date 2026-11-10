"""Generate the SYNTHETIC real-species tree-crown dataset (same layout as demo_tree_crowns.h5).

Usage:
  python scripts/make_real_species_data.py                                  # 2000 crowns, 20 species, 10 sites
  python scripts/make_real_species_data.py --num-samples 1000 --out data/real_species_1k.h5
  python scripts/make_real_species_data.py --sites HARV BART SCBI SERC UNDE  # subset of sites
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):  # Windows consoles/pipes default to cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from src.data.inspect import print_h5_summary  # noqa: E402
from src.data.real_species import generate_real_species_dataset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/real_species_tree_crowns.h5")
    ap.add_argument("--num-samples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--species", nargs="*", default=None, help="taxon codes, e.g. QURU ACRU PIST")
    ap.add_argument("--sites", nargs="*", default=None, help="NEON site codes, e.g. HARV BART")
    a = ap.parse_args()
    t0 = time.time()
    print(f"*** Generating SYNTHETIC crowns with real species names ({a.num_samples}) -> {a.out}")
    p = generate_real_species_dataset(a.out, a.num_samples, a.seed, a.species, a.sites)
    print(f"done in {time.time() - t0:.1f}s, {p.stat().st_size / 1e6:.1f} MB\n")
    print_h5_summary(p, sample_stats=False)
    with h5py.File(p, "r") as f:
        names, codes = list(f.attrs["species_names"]), list(f.attrs["species_codes"])
        sci, sites = list(f.attrs["species_scientific_names"]), list(f.attrs["site_names"])
        sp, st, h = f["species"][:], f["site"][:], f["height"][:]
        print(f"\n{'code':6s} {'species':24s} {'scientific':26s} {'n':>4s} {'height':>7s}  sites")
        for c in np.argsort(-np.bincount(sp, minlength=len(names))):
            m = sp == c
            print(f"{codes[c]:6s} {names[c]:24s} {sci[c]:26s} {m.sum():4d} {h[m].mean():6.1f}m  "
                  + ",".join(sorted({sites[s] for s in st[m]})))
        print("crowns per site:", dict(zip(sites, np.bincount(st, minlength=len(sites)).tolist())))
        print("degraded fraction [rgb, hsi, lidar]:", f["degraded"][:].mean(0).round(3).tolist())


if __name__ == "__main__":
    main()
