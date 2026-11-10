"""Reproduce every Phase-6 result with one command: variants -> model selection on VAL -> ablations -> report.

Usage:
  python scripts/run_experiments.py                       # full plan (~1.5-2 h on a 2-core CPU, faster on GPU)
  python scripts/run_experiments.py --quick               # 3 epochs / 256 train samples: pipeline check (~5 min)
  python scripts/run_experiments.py --only baseline       # a subset of runs

Each run: outputs/experiments/<name>/{best.pt, metrics.jsonl, eval_val/, eval_test/}.
Report:   outputs/experiments/{summary.json, results.md, plots/*.png}.
Finished runs are skipped (delete the folder or pass --force to retrain).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):  # Windows consoles/pipes default to cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(ROOT))

from src.data.synthetic import SyntheticConfig, generate_synthetic_dataset  # noqa: E402
from src.evaluation.evaluate import evaluate  # noqa: E402
from src.evaluation.report import write_report  # noqa: E402
from src.training.trainer import Trainer  # noqa: E402
from src.utils.common import deep_merge, get_logger, load_config  # noqa: E402

log = get_logger("experiments")


def run_one(name: str, overrides: dict[str, Any], base: str, out_root: Path, quick: bool, force: bool,
            device: str) -> dict[str, Any]:
    """Train (unless already done) and evaluate one configuration on val and test sites."""
    out = out_root / name
    ov = deep_merge(overrides, {"paths": {"output_dir": str(out), "norm_stats": str(out / "norm_stats.json")},
                                "train": {"device": device}})
    if quick:
        ov = deep_merge(ov, {"train": {"epochs": 3}, "data": {"max_train_samples": 256}})
    cfg = load_config(base, ov)
    if not Path(cfg["paths"]["data_file"]).exists():
        log.info("data file missing -> generating SYNTHETIC demo data")
        generate_synthetic_dataset(cfg["paths"]["data_file"], SyntheticConfig(**cfg["synthetic"]), seed=cfg["seed"])
    done = (out / "best.pt").exists() and (out / "DONE").exists()
    if force or not done:
        if out.exists() and (out / "metrics.jsonl").exists():
            (out / "metrics.jsonl").unlink()            # fresh log for a fresh run
        t0 = time.time()
        log.info(f"=== training {name}  overrides={json.dumps(overrides)}")
        Trainer(cfg).fit()
        (out / "DONE").write_text(f"{time.time() - t0:.0f}s\n", encoding="utf-8")
    else:
        log.info(f"=== {name}: found finished run, skipping training")
    res = {"name": name, "overrides": overrides, "train_time": (out / "DONE").read_text(encoding="utf-8").strip()}
    for split in ("val", "test"):
        res[split] = evaluate(out / "best.pt", out / f"eval_{split}", split, device=device, make_plots=(split == "test"))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--plan", default="configs/experiments.yaml")
    ap.add_argument("--out", default="outputs/experiments")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    plan = yaml.safe_load(open(ROOT / args.plan, encoding="utf-8"))
    out_root = ROOT / args.out
    out_root.mkdir(parents=True, exist_ok=True)
    base = str(ROOT / args.config)

    results: dict[str, Any] = {}
    for name, ov in plan["variants"].items():
        if args.only is None or name in args.only:
            results[name] = run_one(name, ov or {}, base, out_root, args.quick, args.force, args.device)
    variants = [n for n in plan["variants"] if n in results]
    if not variants:
        sys.exit("no variant run; cannot select a final model")
    sel = max(variants, key=lambda n: results[n]["val"]["scenarios"]["all_sensors"]["all"]["macro_f1"])
    log.info(f"selected final variant by VAL macro-F1: {sel}")
    final_ov = plan["variants"][sel] or {}
    for name, ov in plan["ablations"].items():
        full = f"{sel}__{name}"
        if args.only is None or name in args.only or full in args.only:
            results[full] = run_one(full, deep_merge(final_ov, ov), base, out_root, args.quick, args.force, args.device)
    sc = plan.get("seed_check") or {}
    for seed in sc.get("seeds", []):
        for which in sc.get("runs", []):
            ov = final_ov if which == "final" else deep_merge(final_ov, plan["ablations"][which])
            name = f"{sel}__seed{seed}" if which == "final" else f"{sel}__{which}__seed{seed}"
            if args.only is None or name in args.only:
                results[name] = run_one(name, deep_merge(ov, {"seed": int(seed)}), base, out_root, args.quick,
                                        args.force, args.device)
    summary = {"selected_variant": sel, "selection_metric": "val macro-F1 (all sensors)", "quick": args.quick,
               "runs": results}
    (out_root / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    md = write_report(summary, out_root)
    print(md)


if __name__ == "__main__":
    main()
