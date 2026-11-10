"""Predict tree species, height and crown diameter, and build a visual report.

Inputs (pick one):
  --random 8                      random crowns from the TEST split (never seen in training)   [default]
  --index 12 57 301               specific crowns (rows) of the data file
  --per-species 1                 N test crowns of every species (one card per species)
  --rgb crown.png [--hsi cube.npy] [--lidar chm.npy]
                                  your own crown: any subset of the three sensors
                                    rgb   PNG/JPG/.npy, crown-centred, 16 m across (resized to 32x32)
                                    hsi   .npy, 369 bands x h x w (or a single 369-band spectrum), reflectance
                                    lidar .npy, canopy height raster in metres (resized to 32x32)
Options:
  --drop hsi lidar                pretend these sensors are missing (shows graceful degradation)
  --export-examples DIR           save a few test crowns as PNG/.npy files you can feed back in with --rgb ...
  --out predictions               output folder (report.html, predictions.csv, cards/*.png)

Examples:
  python scripts/predict.py
  python scripts/predict.py --index 5 17 --drop lidar
  python scripts/predict.py --per-species 1 --out predictions_all_species
  python scripts/predict.py --export-examples examples
  python scripts/predict.py --rgb examples/crown_1_rgb.png --hsi examples/crown_1_hsi.npy --lidar examples/crown_1_lidar.npy
"""
from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):  # Windows consoles/pipes default to cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from src.data.crown_dataset import build_demo_split, load_arrays, read_meta  # noqa: E402
from src.evaluation.predict_report import (console_block, context_html, prediction_card, summary,  # noqa: E402
                                           write_csv, write_html)
from src.evaluation.predictor import TreePredictor, load_user_hsi, load_user_lidar, load_user_rgb  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="results/real_species_model.pt")
    ap.add_argument("--data", default=None, help="HDF5 file in the demo_crowns layout (default: the model's)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--random", type=int, default=None)
    g.add_argument("--index", type=int, nargs="+", default=None)
    g.add_argument("--per-species", type=int, default=None, metavar="N",
                   help="N test crowns of EVERY species (e.g. 1 -> one card per species)")
    g.add_argument("--rgb", default=None)
    g.add_argument("--export-examples", default=None, metavar="DIR")
    ap.add_argument("--hsi", default=None)
    ap.add_argument("--lidar", default=None)
    ap.add_argument("--drop", nargs="*", default=[], choices=["rgb", "hsi", "lidar"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="predictions")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-open", action="store_true", help="don't open the report in a browser")
    a = ap.parse_args()

    ck = Path(a.checkpoint)
    if not ck.is_absolute():
        ck = ROOT / ck
    if not ck.exists():
        sys.exit(f"checkpoint {ck} not found. Train one with:\n  python scripts/train.py --config configs/real_species.yaml")
    pred = TreePredictor(ck, a.device)
    cfg = pred.cfg
    if a.data:
        cfg["paths"]["data_file"] = str(Path(a.data).resolve())
    data = Path(cfg["paths"]["data_file"])
    out = ROOT / a.out if not Path(a.out).is_absolute() else Path(a.out)
    (out / "cards").mkdir(parents=True, exist_ok=True)

    custom = a.rgb is not None or a.hsi is not None or a.lidar is not None
    wl = None
    crowns, sources, truth = [], [], []
    if custom:
        s = pred.stats
        rgb = load_user_rgb(a.rgb, s["rgb_shape"]) if a.rgb else None
        hsi = load_user_hsi(a.hsi, s["hsi_shape"]) if a.hsi else None
        chm = load_user_lidar(a.lidar, s["lidar_shape"]) if a.lidar else None
        crowns.append((rgb, hsi, chm))
        sources.append(" + ".join(Path(p).name for p in (a.rgb, a.hsi, a.lidar) if p))
        truth.append(None)
        meta_json = Path(a.rgb or a.hsi or a.lidar).with_name(Path(a.rgb or a.hsi or a.lidar).name.split("_rgb")[0]
                                                              .split("_hsi")[0].split("_lidar")[0] + "_truth.json")
        if meta_json.exists():                                     # files written by --export-examples
            truth[0] = json.loads(meta_json.read_text(encoding="utf-8"))
        if data.exists():
            with h5py.File(data, "r") as f:
                wl = f["wavelengths"][:]
    else:
        if not data.exists():
            sys.exit(f"data file {data} not found. Create it with:\n  python scripts/make_real_species_data.py")
        meta = read_meta(data)
        arr = load_arrays(data)
        with h5py.File(data, "r") as f:
            wl = f["wavelengths"][:]
        test_idx = build_demo_split(cfg, meta).test_idx
        rng = np.random.default_rng(a.seed)
        if a.index is not None:
            rows = a.index
        elif a.per_species:
            rows = []
            for c in range(len(meta["names"])):                      # species order of the dataset
                cand = test_idx[meta["labels"][test_idx] == c]
                rows += rng.choice(cand, min(a.per_species, len(cand)), replace=False).tolist()
        else:
            k = a.random or (6 if a.export_examples else 8)
            rows = sorted(rng.choice(test_idx, min(k, len(test_idx)), replace=False).tolist())
        for r in rows:
            if not 0 <= r < len(meta["labels"]):
                sys.exit(f"--index {r} out of range (0..{len(meta['labels']) - 1})")
            crowns.append((arr["rgb"][r], arr["hsi"][r].astype(np.float32), arr["lidar"][r][0]))
            in_test = "test crown" if r in set(test_idx.tolist()) else "TRAINING crown"
            sources.append(f"row {r} ({in_test}, {meta['site_names'][meta['site'][r]]})")
            truth.append({"species": meta["names"][meta["labels"][r]], "height_m": float(meta["height"][r]),
                          "crown_diameter_m": float(meta["crown"][r])})
        if a.export_examples:
            ex = Path(a.export_examples)
            ex.mkdir(parents=True, exist_ok=True)
            from PIL import Image
            for j, ((rgb, hsi, chm), t) in enumerate(zip(crowns, truth), 1):
                Image.fromarray(np.moveaxis(rgb, 0, -1)).resize((256, 256), Image.NEAREST).save(ex / f"crown_{j}_rgb.png")
                np.save(ex / f"crown_{j}_hsi.npy", hsi.astype(np.float32))
                np.save(ex / f"crown_{j}_lidar.npy", chm.astype(np.float32))
                (ex / f"crown_{j}_truth.json").write_text(json.dumps(t, indent=1), encoding="utf-8")
            print(f"Saved {len(crowns)} example crowns to {ex}/ (crown_N_rgb.png, _hsi.npy, _lidar.npy, _truth.json).")
            print(f"Try:  python scripts/predict.py --rgb {ex}/crown_1_rgb.png --hsi {ex}/crown_1_hsi.npy --lidar {ex}/crown_1_lidar.npy")
            return

    preds = pred.predict(crowns, drop=a.drop, seed=a.seed)
    for p, src, t in zip(preds, sources, truth):
        p.source = src
        if t:
            p.true_species, p.true_height_m, p.true_crown_m = t["species"], t["height_m"], t["crown_diameter_m"]

    print(f"\nModel: {ck.name} (epoch {pred.epoch}), {len(pred.names)} species"
          + (f", sensors dropped: {', '.join(a.drop)}" if a.drop else "") + "\n")
    cards = []
    for i, (p, (rgb, hsi, chm)) in enumerate(zip(preds, crowns)):
        print(console_block(i, p) + "\n")
        cards.append(prediction_card(p, rgb, hsi, chm, wl, out / "cards" / f"crown_{i + 1}.png", f"Crown {i + 1}"))
    if a.per_species:                                                # per-species scoreboard
        print("Per-species result (correct / shown):")
        for name in dict.fromkeys(p.true_species for p in preds):
            ps = [p for p in preds if p.true_species == name]
            ok = sum(p.correct for p in ps)
            print(f"  {name:<24s} {ok}/{len(ps)}  {'OK' if ok == len(ps) else '<- ' + ', '.join(sorted({p.species for p in ps if not p.correct}))}")
        print()
    s = summary(preds)
    if s.get("n_known"):
        print(f"Summary over {s['n_known']} crowns with known truth: species correct {s['accuracy'] * 100:.0f}%, "
              f"true species in top-5 {s['top5'] * 100:.0f}%, height error {s['height_mae']:.1f} m, "
              f"crown-diameter error {s['crown_mae']:.1f} m")
    ctx = ""
    if not custom and not a.drop:                 # whole-test-set reference numbers (fast: one batched pass)
        allp = []
        for k in range(0, len(test_idx), 128):
            rows_k = test_idx[k: k + 128]
            ps = pred.predict([(arr["rgb"][r], arr["hsi"][r].astype(np.float32), arr["lidar"][r][0]) for r in rows_k])
            for p, r in zip(ps, rows_k):
                p.true_species = meta["names"][meta["labels"][r]]
                p.true_height_m, p.true_crown_m = float(meta["height"][r]), float(meta["crown"][r])
            allp += ps
        ts = summary(allp)
        print(f"Whole test set ({len(allp)} crowns): species correct {ts['accuracy'] * 100:.0f}%, "
              f"top-5 {ts['top5'] * 100:.1f}%, height error {ts['height_mae']:.1f} m")
        ctx = context_html(ts, len(allp), "held-out test crowns")
    write_csv(preds, out / "predictions.csv")
    sub = (f"{len(preds)} crown(s) · model trained on {len(pred.names)} species"
           + (" · test crowns were never seen in training" if not custom else " · user-supplied input")
           + (f" · sensors dropped: {', '.join(a.drop)}" if a.drop else ""))
    rep = write_html(preds, cards, out / "report.html", {
        "subtitle": sub, "context_html": ctx})
    print(f"\nReport: {rep}\nCards:  {out / 'cards'}\nTable:  {out / 'predictions.csv'}")
    if not a.no_open:
        try:
            webbrowser.open(rep.resolve().as_uri())
        except Exception:
            pass


if __name__ == "__main__":
    main()
