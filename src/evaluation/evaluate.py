"""Evaluation suite: metrics, calibration, missing-modality robustness, per-species/per-site, trust analysis."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import make_eval_dataset
from src.evaluation.metrics import (confusion_matrix, expected_calibration_error, full_metrics,
                                    per_class_report)
from src.evaluation.predict import predict
from src.models.model import MultimodalTreeNet
from src.utils.common import PROJECT_ROOT, deep_merge, get_device

# keep-mask (rgb, hsi, lidar) for every missing-sensor scenario
SCENARIOS: dict[str, tuple[int, int, int]] = {
    "all_sensors": (1, 1, 1),
    "drop_rgb": (0, 1, 1), "drop_hsi": (1, 0, 1), "drop_lidar": (1, 1, 0),
    "lidar_only": (0, 0, 1), "hsi_only": (0, 1, 0), "rgb_only": (1, 0, 0),
}
MODS = ("rgb", "hsi", "lidar")
# test-time corruptions (sensor present but unreliable): name -> (kind, severity in z-units)
CORRUPTIONS: dict[str, tuple[str, float]] = {
    "rgb_noise": ("rgb_noise", 2.0), "hsi_noise": ("hsi_noise", 2.0), "lidar_noise": ("lidar_noise", 1.0),
}


def load_model(ckpt_path: str | Path, device: torch.device, cfg_overrides: dict | None = None
               ) -> tuple[MultimodalTreeNet, dict[str, Any], dict[str, Any]]:
    """Rebuild the model from a checkpoint (its own config + normalization stats)."""
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = deep_merge(ck["cfg"], cfg_overrides or {})
    if not Path(cfg["paths"]["data_file"]).exists():
        # Checkpoint trained on another machine: use the file of the same name in this project's data/.
        cfg["paths"]["data_file"] = str(PROJECT_ROOT / "data" / Path(cfg["paths"]["data_file"]).name)
    st = ck["norm_stats"]
    sd = ck["model"]
    hsi_bands = sd["encoders.hsi.reduce.0.weight"].shape[1]               # read from weights, not assumed
    lidar_feat = sd["encoders.lidar.point_mlp.0.weight"].shape[1]
    model = MultimodalTreeNet(cfg, ck["num_classes"], hsi_bands, lidar_feat,
                              (st["height_mean"], st["crown_mean"]), (st["height_std"], st["crown_std"]))
    model.load_state_dict(sd)
    return model.to(device).eval(), cfg, ck


def _regression(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    err = pred - true
    ss_tot = ((true - true.mean()) ** 2).sum()
    return {"mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean())),
            "r2": float(1 - (err ** 2).sum() / ss_tot) if ss_tot > 0 else 0.0}


def _trust_analysis(p: dict[str, np.ndarray], degraded: np.ndarray) -> dict[str, dict[str, float]]:
    """Mean trust / fusion weight for clean vs degraded present sensors (synthetic ground truth)."""
    out: dict[str, dict[str, float]] = {}
    for i, m in enumerate(MODS):
        pres = p["mask"][:, i] > 0
        clean, deg = pres & ~degraded[:, i], pres & degraded[:, i]
        out[m] = {"trust_clean": float(p["trust"][clean, i].mean()) if clean.any() else float("nan"),
                  "trust_degraded": float(p["trust"][deg, i].mean()) if deg.any() else float("nan"),
                  "weight_clean": float(p["weights"][clean, i].mean()) if clean.any() else float("nan"),
                  "weight_degraded": float(p["weights"][deg, i].mean()) if deg.any() else float("nan"),
                  "n_clean": int(clean.sum()), "n_degraded": int(deg.sum())}
        # AUROC of (1 - trust) for detecting degraded sensors: P(trust_clean > trust_degraded)
        if clean.any() and deg.any():
            tc, td = p["trust"][clean, i], p["trust"][deg, i]
            out[m]["auroc_degraded_detection"] = float(((tc[:, None] > td[None, :]).mean()
                                                        + 0.5 * (tc[:, None] == td[None, :]).mean()))
    return out


def evaluate(ckpt_path: str | Path, out_dir: str | Path, split: str = "test", cfg_overrides: dict | None = None,
             make_plots: bool = True, device: str = "auto", scenarios: dict[str, tuple[int, int, int]] | None = None
             ) -> dict[str, Any]:
    """Evaluate a checkpoint on the val or test sites and write results to ``out_dir``.

    Files written: results.json, per_species.csv, confusion_matrix.csv, per_site.csv, plots/*.png.
    Metrics are reported on ALL samples of the split and on the SEEN subset (species that occur in
    the training sites) — species unseen in training cannot be predicted correctly by construction.
    """
    out_dir = Path(out_dir)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    dev = get_device(device)
    model, cfg, ck = load_model(ckpt_path, dev, cfg_overrides)
    ds, sp, info = make_eval_dataset(cfg, ck["norm_stats"], split)
    loader = DataLoader(ds, batch_size=int(cfg["data"]["batch_size"]), shuffle=False, num_workers=0)
    C = ck["num_classes"]
    names, all_labels, degraded_all, is_synth = info["names"], info["labels"], info["degraded"], info["synthetic"]
    train_counts = np.bincount(all_labels[sp.train_idx], minlength=C)

    res: dict[str, Any] = {"checkpoint": str(ckpt_path), "checkpoint_epoch": int(ck["epoch"]), "split": split,
                           "sites": sp.test_sites if split == "test" else sp.val_sites, "synthetic_data": is_synth,
                           "n_samples": int(len(ds)), "scenarios": {}}
    preds: dict[str, dict[str, np.ndarray]] = {}
    for name, keep in (scenarios or SCENARIOS).items():
        p = predict(model, loader, dev, keep)
        preds[name] = p
        seen = train_counts[p["label"]] > 0
        res["scenarios"][name] = {
            "keep": list(keep),
            "all": full_metrics(p["label"], p["prob"], C),
            "seen_species": full_metrics(p["label"][seen], p["prob"][seen], C),
            "mean_fusion_weight": {m: float(p["weights"][:, i].mean()) for i, m in enumerate(MODS)},
            "frac_samples_with_no_sensor": float((p["mask"].sum(1) == 0).mean()),
        }
    res["corruptions"] = {}
    for name, (kind, sev) in CORRUPTIONS.items():
        p = predict(model, loader, dev, None, corrupt=kind, severity=sev, seed=cfg["seed"])
        seen = train_counts[p["label"]] > 0
        res["corruptions"][name] = {
            "kind": kind, "severity": sev,
            "all": full_metrics(p["label"], p["prob"], C),
            "seen_species": full_metrics(p["label"][seen], p["prob"][seen], C),
            "mean_fusion_weight": {m: float(p["weights"][:, i].mean()) for i, m in enumerate(MODS)},
        }
    if degraded_all is not None:   # accuracy on crowns with a (generator-)degraded sensor vs fully clean ones
        p = preds["all_sensors"]
        d = degraded_all[p["index"]].any(1)
        res["degraded_subset"] = {"n_degraded": int(d.sum()),
                                  "acc_degraded": float((p["pred"][d] == p["label"][d]).mean()) if d.any() else float("nan"),
                                  "acc_clean": float((p["pred"][~d] == p["label"][~d]).mean())}
    base = res["scenarios"]["all_sensors"]
    for name, r in list(res["scenarios"].items()) + list(res["corruptions"].items()):
        for sub in ("all", "seen_species"):
            r[sub]["accuracy_drop"] = base[sub]["accuracy"] - r[sub]["accuracy"]
            r[sub]["macro_f1_drop"] = base[sub]["macro_f1"] - r[sub]["macro_f1"]

    p = preds["all_sensors"]
    rep = per_class_report(p["label"], p["pred"], C, names, train_counts)
    with open(out_dir / "per_species.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rep[0].keys()))
        w.writeheader()
        w.writerows(sorted(rep, key=lambda r: -r["support"]))
    cm = confusion_matrix(p["label"], p["pred"], C)
    present = np.where((cm.sum(1) + cm.sum(0)) > 0)[0]
    with open(out_dir / "confusion_matrix.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        lab = [names[c] if names else str(c) for c in present]
        w.writerow(["true\\pred"] + lab)
        for c, l in zip(present, lab):
            w.writerow([l] + cm[c, present].tolist())

    site_rows = []
    for s in np.unique(p["site"]):
        m = p["site"] == s
        r = full_metrics(p["label"][m], p["prob"][m], C)
        site_rows.append({"site": ds.site_names[s], "n": int(m.sum()), "accuracy": r["accuracy"],
                          "macro_f1": r["macro_f1"], "ece": r["ece"]})
    with open(out_dir / "per_site.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(site_rows[0].keys()))
        w.writeheader()
        w.writerows(site_rows)
    res["per_site"] = site_rows
    res["structure"] = {"height_m": _regression(p["structure"][:, 0], p["struct_true"][:, 0]),
                        "crown_diameter_m": _regression(p["structure"][:, 1], p["struct_true"][:, 1])}
    ece, bins = expected_calibration_error(p["prob"], p["label"])
    res["calibration_bins"] = {k: v.tolist() for k, v in bins.items()}
    res["species_with_f1_zero"] = int(sum(r["f1"] == 0 and r["support"] > 0 for r in rep))
    res["test_species_unseen_in_train"] = int(sum(r["train_count"] == 0 and r["support"] > 0 for r in rep))
    if degraded_all is not None:
        res["trust_vs_degradation_synthetic_truth"] = _trust_analysis(p, degraded_all[p["index"]])

    (out_dir / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    if make_plots:
        from src.evaluation import plots
        plots.make_all(res, rep, cm, present, train_counts, p, out_dir / "plots", names)
    return res


def summary_table(res: dict[str, Any]) -> str:
    """Markdown table of the missing-modality scenarios."""
    lines = ["| scenario | acc | macro-F1 | weighted-F1 | ECE | acc drop | acc (seen sp.) | macro-F1 (seen sp.) |",
             "|---|---|---|---|---|---|---|---|"]
    for name, r in res["scenarios"].items():
        a, s = r["all"], r["seen_species"]
        lines.append(f"| {name} | {a['accuracy']:.3f} | {a['macro_f1']:.3f} | {a['weighted_f1']:.3f} | {a['ece']:.3f} "
                     f"| {a['accuracy_drop']:+.3f} | {s['accuracy']:.3f} | {s['macro_f1']:.3f} |")
    return "\n".join(lines)
