"""Generate the SYNTHETIC demo HDF5 dataset (NEON-MMTC-like layout). Not real data.

Usage: python scripts/make_synthetic_data.py --config configs/default.yaml [--num-samples N] [--out PATH]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):  # Windows consoles/pipes default to cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.inspect import print_h5_summary  # noqa: E402
from src.data.synthetic import SyntheticConfig, generate_synthetic_dataset  # noqa: E402
from src.utils.common import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--num-samples", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    scfg = SyntheticConfig(**cfg["synthetic"])
    if args.num_samples:
        scfg.num_samples = args.num_samples
    out = args.out or cfg["paths"]["data_file"]
    t0 = time.time()
    print(f"*** Generating SYNTHETIC demo data ({scfg.num_samples} crowns) -> {out}")
    generate_synthetic_dataset(out, scfg, seed=cfg["seed"])
    print(f"done in {time.time() - t0:.1f}s, {Path(out).stat().st_size / 1e6:.1f} MB\n")
    print_h5_summary(out)


if __name__ == "__main__":
    main()
