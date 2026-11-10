"""Static evaluation plots (matplotlib, PNG). Palette: validated categorical slots 1-2 + a blue sequential ramp."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S1, S2 = "#2a78d6", "#eb6834"           # categorical slots 1, 2 (validated: CVD ΔE 24.7)
BLUES = LinearSegmentedColormap.from_list("seq_blue", [SURFACE, "#cde2fb", "#86b6ef", "#2a78d6", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 10, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False,
})


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_missing_modality(res: dict[str, Any], path: Path) -> None:
    """Grouped horizontal bars: accuracy and macro-F1 per sensor scenario (same 0-1 scale)."""
    names = list(res["scenarios"])
    acc = [res["scenarios"][n]["all"]["accuracy"] for n in names]
    f1 = [res["scenarios"][n]["all"]["macro_f1"] for n in names]
    y = np.arange(len(names))[::-1]
    h = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.barh(y + h / 2, acc, h * 0.92, color=S1, label="Accuracy")
    ax.barh(y - h / 2, f1, h * 0.92, color=S2, label="Macro-F1")
    for yi, a, f in zip(y, acc, f1):
        ax.text(a + 0.01, yi + h / 2, f"{a:.2f}", va="center", color=INK2, fontsize=8)
        ax.text(f + 0.01, yi - h / 2, f"{f:.2f}", va="center", color=INK2, fontsize=8)
    ax.set_yticks(y, [n.replace("_", " ") for n in names])
    ax.set_xlim(0, max(1.0, max(acc) + 0.1))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Score on held-out test sites")
    ax.set_title("Robustness to missing sensors")
    ax.legend(loc="lower right")
    _save(fig, path)


def plot_reliability(res: dict[str, Any], path: Path) -> None:
    """Reliability diagram: per-bin accuracy vs mean confidence, with the identity line."""
    b = res["calibration_bins"]
    edges = np.array(b["edges"])
    cnt = np.array(b["count"])
    centers = (edges[:-1] + edges[1:]) / 2
    w = edges[1] - edges[0]
    ok = cnt > 0
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--", label="Perfect calibration")
    ax.bar(centers[ok], np.array(b["accuracy"])[ok], w * 0.9, color=S1, label="Observed accuracy")
    ax.plot(np.array(b["confidence"])[ok], np.array(b["accuracy"])[ok], "o", ms=4, color=INK, label="Bin mean confidence")
    ece = res["scenarios"]["all_sensors"]["all"]["ece"]
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Confidence", ylabel="Accuracy")
    ax.set_title(f"Calibration (ECE = {ece:.3f})")
    ax.legend(loc="upper left", fontsize=8)
    _save(fig, path)


def plot_confusion(cm: np.ndarray, present: np.ndarray, train_counts: np.ndarray, path: Path) -> None:
    """Row-normalized confusion matrix over classes in the split, ordered by training frequency."""
    order = present[np.argsort(-train_counts[present], kind="stable")]
    sub = cm[np.ix_(order, order)].astype(float)
    rows = sub.sum(1, keepdims=True)
    norm = np.divide(sub, rows, out=np.zeros_like(sub), where=rows > 0)
    fig, ax = plt.subplots(figsize=(7, 6.2))
    im = ax.imshow(norm, cmap=BLUES, vmin=0, vmax=1, interpolation="nearest")
    ax.grid(False)
    ax.set_xlabel("Predicted species (sorted by training count →)")
    ax.set_ylabel("True species")
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(f"Confusion matrix, row-normalized\n({len(order)} species that are true or predicted in the split)", fontsize=11)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Fraction of true class", color=INK2)
    cb.outline.set_visible(False)
    _save(fig, path)


def plot_f1_vs_train_count(rep: list[dict], path: Path) -> None:
    """Per-species test F1 against the number of training crowns (log x)."""
    tc = np.array([r["train_count"] for r in rep if r["support"] > 0])
    f1 = np.array([r["f1"] for r in rep if r["support"] > 0])
    sup = np.array([r["support"] for r in rep if r["support"] > 0])
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    seen = tc > 0
    ax.scatter(tc[seen], f1[seen], s=12 + 4 * sup[seen], color=S1, alpha=0.75, edgecolor=SURFACE, lw=0.8,
               label="Species seen in training")
    if (~seen).any():
        ax.scatter(np.full((~seen).sum(), 0.7), f1[~seen], s=12 + 4 * sup[~seen], color=S2, alpha=0.9,
                   edgecolor=SURFACE, lw=0.8, marker="s",
                   label=f"Unseen in training: {(~seen).sum()} species, F1 = 0 by construction")
    ax.set_xscale("log")
    ax.set(xlabel="Training crowns of the species (log scale)", ylabel="Test F1", ylim=(-0.03, 1.03))
    ax.set_title("Per-species F1 vs training frequency (marker size = test support)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8, markerscale=0.5)
    _save(fig, path)


def plot_trust(res: dict[str, Any], path: Path) -> None:
    """Mean fusion weight for clean vs degraded sensors (synthetic ground truth)."""
    t = res.get("trust_vs_degradation_synthetic_truth")
    if not t:
        return
    mods = list(t)
    clean = [t[m]["weight_clean"] for m in mods]
    deg = [t[m]["weight_degraded"] for m in mods]
    x = np.arange(len(mods))
    w = 0.36
    fig, ax = plt.subplots(figsize=(5.6, 4))
    ax.bar(x - w / 2, clean, w * 0.92, color=S1, label="Clean sensor")
    ax.bar(x + w / 2, deg, w * 0.92, color=S2, label="Degraded sensor")
    for xi, c, d in zip(x, clean, deg):
        ax.text(xi - w / 2, c + 0.01, f"{c:.2f}", ha="center", color=INK2, fontsize=8)
        ax.text(xi + w / 2, d + 0.01, f"{d:.2f}", ha="center", color=INK2, fontsize=8)
    ax.set_xticks(x, [f"{m.upper()}\n(AUROC {t[m].get('auroc_degraded_detection', float('nan')):.2f})" for m in mods])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Mean fusion weight")
    ax.set_title("Fusion weight: clean vs degraded sensors")
    ax.legend(loc="upper right", fontsize=8)
    _save(fig, path)


def make_all(res: dict[str, Any], rep: list[dict], cm: np.ndarray, present: np.ndarray, train_counts: np.ndarray,
             preds: dict[str, np.ndarray], out: Path, names: list[str] | None = None) -> list[Path]:
    """Write every plot into ``out``; returns the paths."""
    out.mkdir(parents=True, exist_ok=True)
    plot_missing_modality(res, out / "missing_modality.png")
    plot_reliability(res, out / "reliability.png")
    plot_confusion(cm, present, train_counts, out / "confusion_matrix.png")
    plot_f1_vs_train_count(rep, out / "per_species_f1.png")
    plot_trust(res, out / "trust_weights.png")
    return sorted(out.glob("*.png"))
