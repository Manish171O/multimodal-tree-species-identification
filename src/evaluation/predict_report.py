"""Presentation of predictions: console summary, one PNG "prediction card" per crown, CSV and an HTML report."""
from __future__ import annotations

import base64
import csv
import html
from pathlib import Path
from typing import Sequence

import numpy as np

from src.evaluation import plots as P
from src.evaluation.predictor import Prediction

plt = P.plt
GOOD, BAD = "#0ca30c", "#d03b3b"            # status colours (always paired with a text label)


# ------------------------------------------------------------------------------------- console
def console_block(i: int, p: Prediction) -> str:
    """Readable multi-line summary of one prediction."""
    head = f"Crown {i + 1}  [{p.source}]"
    if p.correct is not None:
        head += "   ->  " + ("CORRECT" if p.correct else f"WRONG (true: {p.true_species})")
    lines = [head, "-" * len(head)]
    for r, (name, sci, pr) in enumerate(p.top_species):
        bar = "#" * int(round(pr * 30))
        mark = "  <- true species" if name == p.true_species else ""
        lines.append(f"  {r + 1}. {name:<22s} {sci:<26s} {pr * 100:5.1f}%  {bar}{mark}")
    t_h = f"   (true {p.true_height_m:.1f} m)" if p.true_height_m is not None else ""
    t_c = f"   (true {p.true_crown_m:.1f} m)" if p.true_crown_m is not None else ""
    lines.append(f"  Height:          {p.height_m:5.1f} m{t_h}")
    lines.append(f"  Crown diameter:  {p.crown_diameter_m:5.1f} m{t_c}")
    lines.append("  Sensor trust:    " + "   ".join(
        f"{m.upper()} {p.trust[m]:.2f}" if m in p.sensors_used else f"{m.upper()} missing" for m in ("rgb", "hsi", "lidar")))
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------- card
def prediction_card(p: Prediction, rgb: np.ndarray | None, hsi: np.ndarray | None, chm: np.ndarray | None,
                    wavelengths: np.ndarray | None, path: Path, title: str) -> Path:
    """One PNG: the three sensor views, top-5 species bars, height / crown and sensor weights."""
    fig = plt.figure(figsize=(11, 5.4))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 0.95], hspace=0.55, wspace=0.9)
    ax_rgb, ax_lid, ax_spec = fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[0, 4:6])
    ax_top, ax_info = fig.add_subplot(gs[1, 0:4]), fig.add_subplot(gs[1, 4:6])

    def missing(ax, label):
        ax.set_facecolor("#f0efec"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.text(0.5, 0.5, f"{label}\nnot provided", ha="center", va="center", color=P.INK2, transform=ax.transAxes)

    if rgb is not None and "rgb" in p.sensors_used:
        img = np.moveaxis(rgb, 0, -1).astype(np.float32)
        ax_rgb.imshow(np.clip(img / max(np.percentile(img, 99), 1), 0, 1), interpolation="nearest")
        ax_rgb.set_xticks([]); ax_rgb.set_yticks([]); ax_rgb.grid(False)
    else:
        missing(ax_rgb, "RGB image")
    ax_rgb.set_title("RGB camera", fontsize=10)
    if chm is not None and "lidar" in p.sensors_used:
        im = ax_lid.imshow(chm, cmap=P.BLUES, vmin=0, vmax=max(float(np.max(chm)), 1.0), interpolation="nearest")
        ax_lid.set_xticks([]); ax_lid.set_yticks([]); ax_lid.grid(False)
        cb = fig.colorbar(im, ax=ax_lid, fraction=0.046, pad=0.03); cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7)
    else:
        missing(ax_lid, "LiDAR")
    ax_lid.set_title("LiDAR canopy height (m)", fontsize=10)
    if hsi is not None and "hsi" in p.sensors_used:
        h, w = hsi.shape[-2:]
        spec = hsi[:, h // 2 - 1: h // 2 + 1, w // 2 - 1: w // 2 + 1].mean((1, 2))
        x = wavelengths if wavelengths is not None and len(wavelengths) == len(spec) else np.arange(len(spec))
        ax_spec.plot(x, spec, color=P.S1, lw=1.4)
        ax_spec.set_xlabel("wavelength (nm)", fontsize=8); ax_spec.set_ylabel("reflectance", fontsize=8)
        ax_spec.tick_params(labelsize=7); ax_spec.set_ylim(0, max(0.7, float(spec.max()) * 1.1))
    else:
        missing(ax_spec, "Hyperspectral")
    ax_spec.set_title("Hyperspectral (crown centre)", fontsize=10)

    names = [f"{n}  ({s})" for n, s, _ in p.top_species][::-1]
    probs = [pr for _, _, pr in p.top_species][::-1]
    cols = [P.S1 if n == p.species else "#b7d3f6" for n, _, _ in p.top_species][::-1]
    y = np.arange(len(names))
    ax_top.barh(y, probs, 0.65, color=cols)
    ax_top.set_yticks(y, names, fontsize=9)
    for yi, (n, _, pr) in zip(y, list(p.top_species)[::-1]):
        lab = f"{pr * 100:.1f}%" + ("   ← true species" if n == p.true_species else "")
        ax_top.text(pr + 0.01, yi, lab, va="center", fontsize=8.5, color=P.INK2)
    ax_top.set_xlim(0, 1.18); ax_top.grid(axis="y", visible=False)
    ax_top.set_xlabel("predicted probability", fontsize=8); ax_top.tick_params(axis="x", labelsize=7)
    ax_top.set_title("Top-5 species", fontsize=10, loc="left")

    ax_info.axis("off")
    rows = [("Height", f"{p.height_m:.1f} m", None if p.true_height_m is None else f"true {p.true_height_m:.1f} m"),
            ("Crown diameter", f"{p.crown_diameter_m:.1f} m", None if p.true_crown_m is None else f"true {p.true_crown_m:.1f} m")]
    yy = 0.95
    for k, v, t in rows:
        ax_info.text(0.0, yy, k, fontsize=9, color=P.INK2, transform=ax_info.transAxes, va="top")
        ax_info.text(0.0, yy - 0.11, v, fontsize=15, color=P.INK, weight="bold", transform=ax_info.transAxes, va="top")
        if t:
            ax_info.text(0.55, yy - 0.13, t, fontsize=9, color=P.INK2, transform=ax_info.transAxes, va="top")
        yy -= 0.33
    ax_info.text(0.0, yy + 0.02, "Fusion weight per sensor", fontsize=9, color=P.INK2, transform=ax_info.transAxes, va="top")
    for j, m in enumerate(("rgb", "hsi", "lidar")):
        yb = yy - 0.16 - j * 0.11
        wv = p.weight[m] if m in p.sensors_used else 0.0
        ax_info.add_patch(plt.Rectangle((0.28, yb - 0.035), 0.6, 0.06, color="#e4e3df", transform=ax_info.transAxes, clip_on=False))
        ax_info.add_patch(plt.Rectangle((0.28, yb - 0.035), 0.6 * wv, 0.06, color=P.S1, transform=ax_info.transAxes, clip_on=False))
        ax_info.text(0.0, yb, m.upper(), fontsize=8, color=P.INK2, transform=ax_info.transAxes, va="center")
        ax_info.text(0.9, yb, f"{wv:.2f}" if m in p.sensors_used else "missing", fontsize=8, color=P.INK2,
                     transform=ax_info.transAxes, va="center")

    verdict = ""
    if p.correct is not None:
        verdict = "  ✔ correct" if p.correct else f"  ✘ wrong (true: {p.true_species})"
    fig.suptitle(f"{title}:  {p.species} ({p.top_species[0][1]}), {p.confidence * 100:.0f}% confident", x=0.02,
                 ha="left", fontsize=13, weight="bold", y=0.995)
    if verdict:
        fig.text(0.98, 0.965, verdict.strip(), ha="right", fontsize=11, weight="bold",
                 color=GOOD if p.correct else BAD)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


# ------------------------------------------------------------------------------------- csv/html
def write_csv(preds: Sequence[Prediction], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["crown", "source", "predicted_species", "scientific_name", "confidence", "true_species", "correct",
                    "pred_height_m", "true_height_m", "pred_crown_m", "true_crown_m", "sensors_used",
                    "trust_rgb", "trust_hsi", "trust_lidar", "top5"])
        for i, p in enumerate(preds):
            w.writerow([i + 1, p.source, p.species, p.top_species[0][1], f"{p.confidence:.4f}", p.true_species or "",
                        "" if p.correct is None else p.correct, f"{p.height_m:.2f}",
                        "" if p.true_height_m is None else f"{p.true_height_m:.2f}", f"{p.crown_diameter_m:.2f}",
                        "" if p.true_crown_m is None else f"{p.true_crown_m:.2f}", "+".join(p.sensors_used),
                        f"{p.trust['rgb']:.3f}", f"{p.trust['hsi']:.3f}", f"{p.trust['lidar']:.3f}",
                        "; ".join(f"{n} {pr:.3f}" for n, _, pr in p.top_species)])


def summary(preds: Sequence[Prediction]) -> dict[str, float | int | None]:
    """Accuracy / errors over the predictions that have ground truth."""
    known = [p for p in preds if p.true_species is not None]
    if not known:
        return {"n": len(preds), "n_known": 0}
    return {"n": len(preds), "n_known": len(known),
            "accuracy": float(np.mean([p.correct for p in known])),
            "top5": float(np.mean([p.true_species in [n for n, _, _ in p.top_species] for p in known])),
            "height_mae": float(np.mean([abs(p.height_m - p.true_height_m) for p in known])),
            "crown_mae": float(np.mean([abs(p.crown_diameter_m - p.true_crown_m) for p in known])),
            "mean_conf": float(np.mean([p.confidence for p in known]))}


def write_html(preds: Sequence[Prediction], cards: Sequence[Path], path: Path, meta: dict[str, str]) -> Path:
    """Self-contained HTML report (images embedded) to open in any browser."""
    s = summary(preds)
    tiles = [("Crowns predicted", f"{s['n']}")]
    if s.get("n_known"):
        tiles += [("Species correct", f"{s['accuracy'] * 100:.0f}%"), ("True species in top 5", f"{s['top5'] * 100:.0f}%"),
                  ("Height error (mean)", f"{s['height_mae']:.1f} m"), ("Crown-diameter error (mean)", f"{s['crown_mae']:.1f} m")]
    rows = ""
    for i, p in enumerate(preds):
        status = "" if p.correct is None else (
            '<span class="ok">✔ correct</span>' if p.correct else '<span class="bad">✘ wrong</span>')
        rows += (f"<tr><td>{i + 1}</td><td>{html.escape(p.source)}</td><td><b>{html.escape(p.species)}</b>"
                 f"<div class=sci>{html.escape(p.top_species[0][1])}</div></td><td>{p.confidence * 100:.0f}%</td>"
                 f"<td>{html.escape(p.true_species or '–')}</td><td>{status}</td>"
                 f"<td>{p.height_m:.1f}" + ("" if p.true_height_m is None else f" <span class=t>/ {p.true_height_m:.1f}</span>")
                 + f"</td><td>{p.crown_diameter_m:.1f}" + ("" if p.true_crown_m is None else f" <span class=t>/ {p.true_crown_m:.1f}</span>")
                 + f"</td><td>{', '.join(m.upper() for m in p.sensors_used)}</td></tr>")
    imgs = "".join(f'<figure><img src="data:image/png;base64,{base64.b64encode(c.read_bytes()).decode()}" '
                   f'alt="Prediction card {i + 1}"></figure>' for i, c in enumerate(cards))
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Tree Species Predictions</title>
<style>
:root {{ --bg:#f6f8f5; --fg:#16201a; --muted:#56615a; --rule:#d9dfd8; --panel:#fff; --ok:#0b7d38; --bad:#b4382c; --warn-bg:#fff6e0; --warn-fg:#6b4a00; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#111613; --fg:#e6ece7; --muted:#a3ada6; --rule:#2a332d; --panel:#18201b; --ok:#4cc27a; --bad:#f08a7e; --warn-bg:#2e2612; --warn-fg:#f1d48c; color-scheme:dark; }} }}
* {{ box-sizing:border-box }} body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif }}
.wrap {{ max-width:1100px; margin:0 auto; padding:32px max(16px,env(safe-area-inset-left)) 64px; display:grid; gap:28px }}
h1 {{ margin:0; font-size:clamp(1.6rem,3.5vw,2.3rem); line-height:1.15 }} .sub {{ color:var(--muted); margin:6px 0 0 }}
.notice {{ background:var(--warn-bg); color:var(--warn-fg); padding:10px 14px; border-radius:6px; font-size:.92rem }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:1px; background:var(--rule); border:1px solid var(--rule); border-radius:8px; overflow:hidden }}
.tile {{ background:var(--panel); padding:14px 16px }} .tile .k {{ color:var(--muted); font-size:.85rem }} .tile .v {{ font-size:1.7rem; font-weight:600; font-variant-numeric:tabular-nums }}
.tw {{ overflow-x:auto; border:1px solid var(--rule); border-radius:8px; background:var(--panel) }}
table {{ border-collapse:collapse; width:100%; font-size:.9rem }} th,td {{ padding:8px 12px; border-bottom:1px solid var(--rule); text-align:left; white-space:nowrap }}
th {{ color:var(--muted); font-weight:600; font-size:.8rem }} td {{ font-variant-numeric:tabular-nums }} tbody tr:last-child td {{ border-bottom:0 }}
.sci {{ color:var(--muted); font-style:italic; font-size:.82rem }} .t {{ color:var(--muted) }} .ok {{ color:var(--ok); font-weight:600 }} .bad {{ color:var(--bad); font-weight:600 }}
figure {{ margin:0; background:#fcfcfb; border:1px solid var(--rule); border-radius:8px; padding:8px }} figure img {{ display:block; width:100%; height:auto }}
h2 {{ margin:0; font-size:1.25rem }} .cards {{ display:grid; gap:16px }}
.context {{ margin:-12px 0 0; color:var(--muted); font-size:.92rem }} .context b {{ color:var(--fg) }}
</style></head><body><div class="wrap">
<header><h1>Tree species predictions</h1>
<p class="sub">{html.escape(meta.get('subtitle', ''))}</p></header>
{f'<p class="notice">{html.escape(meta["notice"])}</p>' if meta.get("notice") else ""}
<section class="tiles">{''.join(f'<div class=tile><div class=k>{k}</div><div class=v>{v}</div></div>' for k, v in tiles)}</section>
{meta.get('context_html', '')}
<section><h2>All predictions</h2><p class="sub">Height and crown columns: predicted / true, in metres.</p>
<div class="tw"><table><thead><tr><th>#</th><th>Input</th><th>Predicted species</th><th>Confidence</th><th>True species</th><th>Result</th><th>Height</th><th>Crown Ø</th><th>Sensors</th></tr></thead>
<tbody>{rows}</tbody></table></div></section>
<section class="cards"><h2>Prediction cards</h2>{imgs}</section>
</div></body></html>"""
    path.write_text(doc, encoding="utf-8")
    return path


def context_html(stats: dict[str, float], n: int, label: str) -> str:
    """One-line reference figures for the whole test set, so a small random sample isn't over-read."""
    return (f'<p class="context">For reference, on <b>all {n} {label}</b> this model gets the species right '
            f'<b>{stats["accuracy"] * 100:.0f}%</b> of the time (true species in top 5: <b>{stats["top5"] * 100:.1f}%</b>), '
            f'with a mean height error of <b>{stats["height_mae"]:.1f} m</b> and crown-diameter error of '
            f'<b>{stats["crown_mae"]:.2f} m</b>. A small random sample can land above or below this.</p>')
