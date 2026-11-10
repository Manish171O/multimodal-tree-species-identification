"""Evaluate a trained checkpoint on the held-out sites.

Usage:
  python scripts/evaluate.py --checkpoint outputs/best.pt                 # test sites -> outputs/eval_test/
  python scripts/evaluate.py --checkpoint outputs/best.pt --split val --out outputs/eval_val

Writes results.json, per_species.csv, per_site.csv, confusion_matrix.csv and plots/*.png, and prints
a missing-modality summary table.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):  # Windows consoles/pipes default to cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.evaluate import evaluate, summary_table  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="outputs/best.pt")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--data", default=None, help="HDF5 data file (default: the checkpoint's, else configs/default.yaml)")
    args = ap.parse_args()
    out = args.out or str(Path(args.checkpoint).parent / f"eval_{args.split}")
    ov = {"paths": {"data_file": str(Path(args.data).resolve())}} if args.data else None
    res = evaluate(args.checkpoint, out, args.split, cfg_overrides=ov, make_plots=not args.no_plots, device=args.device)
    if res["synthetic_data"]:
        print("*** NOTE: evaluated on SYNTHETIC demo data — numbers validate the pipeline, not real-world accuracy.")
    print(f"{args.split} sites {res['sites']}  n={res['n_samples']}  checkpoint epoch {res['checkpoint_epoch']}\n")
    print(summary_table(res))
    a = res["scenarios"]["all_sensors"]["all"]
    print(f"\nprecision {a['macro_precision']:.3f}  recall {a['macro_recall']:.3f}  top-5 {a['top5_accuracy']:.3f}  "
          f"NLL {a['nll']:.3f}")
    print("structure:", json.dumps(res["structure"]))
    if "trust_vs_degradation_synthetic_truth" in res:
        for m, t in res["trust_vs_degradation_synthetic_truth"].items():
            print(f"trust {m:5s} clean {t['trust_clean']:.3f} degraded {t['trust_degraded']:.3f} | weight "
                  f"{t['weight_clean']:.3f} -> {t['weight_degraded']:.3f} | AUROC {t.get('auroc_degraded_detection', float('nan')):.3f}")
    print(f"\nwrote {out}/")


if __name__ == "__main__":
    main()
