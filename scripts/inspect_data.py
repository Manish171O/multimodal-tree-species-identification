"""Print groups, keys, shapes, dtypes and value checks of an HDF5 file, then validate the schema.

Usage: python scripts/inspect_data.py [--file PATH] [--config configs/default.yaml]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):  # Windows consoles/pipes default to cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.inspect import print_h5_summary, validate_schema  # noqa: E402
from src.utils.common import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--file", default=None)
    args = ap.parse_args()
    path = args.file or load_config(args.config)["paths"]["data_file"]
    if not Path(path).exists():
        sys.exit(f"{path} not found. Real data absent -> run scripts/make_synthetic_data.py for SYNTHETIC demo data.")
    print_h5_summary(path)
    shapes = validate_schema(path)
    print("\nSchema OK:", {k: v for k, v in shapes.items()})


if __name__ == "__main__":
    main()
