"""Train the adaptive quality-aware multimodal model.

Usage:
  python scripts/train.py --config configs/default.yaml
  python scripts/train.py --config configs/default.yaml --set train.epochs=5 data.max_train_samples=64
  python scripts/train.py --config configs/default.yaml --resume outputs/last.pt

If the configured data file does not exist, the SYNTHETIC demo dataset is generated first.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

if hasattr(sys.stdout, "reconfigure"):  # Windows consoles/pipes default to cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.synthetic import SyntheticConfig, generate_synthetic_dataset  # noqa: E402
from src.training.trainer import Trainer  # noqa: E402
from src.utils.common import load_config  # noqa: E402


def parse_overrides(pairs: list[str]) -> dict[str, Any]:
    """['a.b=1', 'c=x'] -> {'a': {'b': 1}, 'c': 'x'} (values parsed as YAML)."""
    out: dict[str, Any] = {}
    for p in pairs:
        key, _, val = p.partition("=")
        node = out
        *parents, leaf = key.split(".")
        for k in parents:
            node = node.setdefault(k, {})
        node[leaf] = yaml.safe_load(val)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. train.epochs=5")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, parse_overrides(args.set))
    data = Path(cfg["paths"]["data_file"])
    if not data.exists():
        print(f"*** {data} not found -> generating SYNTHETIC demo data (not real NEON data)")
        generate_synthetic_dataset(data, SyntheticConfig(**cfg["synthetic"]), seed=cfg["seed"])
    trainer = Trainer(cfg)
    if args.resume:
        trainer.resume(args.resume)
    hist = trainer.fit()
    best = max(hist, key=lambda r: r[cfg["train"]["select_metric"]]) if hist else {}
    print(f"best epoch {best.get('epoch')} {cfg['train']['select_metric']}={best.get(cfg['train']['select_metric'], 0):.4f}"
          f" -> {trainer.out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
