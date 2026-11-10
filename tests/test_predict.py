"""demo_crowns loader (CHM -> points, splits, stats) and the predict path (TreePredictor, user files, report)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.crown_dataset import build_demo_split, chm_to_points, read_meta
from src.data.dataset import build_datasets
from src.data.real_species import generate_real_species_dataset
from src.evaluation.predictor import TreePredictor, load_user_hsi, load_user_lidar, load_user_rgb
from src.training.trainer import Trainer
from src.utils.common import PROJECT_ROOT, deep_merge, load_config


@pytest.fixture(scope="module")
def rs_cfg(tmp_path_factory):
    d = tmp_path_factory.mktemp("rsp")
    data = generate_real_species_dataset(d / "rs.h5", num_samples=400, seed=11)
    cfg = load_config(PROJECT_ROOT / "configs" / "real_species.yaml", overrides={
        "paths": {"data_file": str(data), "output_dir": str(d / "out"), "norm_stats": str(d / "out" / "ns.json")},
        "data": {"batch_size": 16, "max_train_samples": 96},
        "train": {"epochs": 1, "device": "cpu", "warmup_epochs": 0}})
    return cfg


@pytest.fixture(scope="module")
def rs_ckpt(rs_cfg):
    tr = Trainer(rs_cfg)
    tr.fit()
    return tr.out_dir / "best.pt"


def test_chm_to_points_skips_no_return_pixels() -> None:
    chm = np.zeros((4, 4), np.float32)
    chm[1, 2], chm[2, 2] = 10.0, -0.3           # a return and a slightly-negative ground return
    pts = chm_to_points(chm, 0.5)
    assert pts.shape == (2, 4) and np.allclose(sorted(pts[:, 2]), [-0.3, 10.0])
    assert np.allclose(np.abs(pts[:, :2]).max(), 0.25)            # pixel centres in metres from the centre


@pytest.mark.parametrize("mode", ["stratified", "site"])
def test_demo_splits(rs_cfg, mode) -> None:
    cfg = deep_merge(rs_cfg, {"split": {"mode": mode}})
    sp = build_demo_split(cfg)
    all_idx = np.concatenate([sp.train_idx, sp.val_idx, sp.test_idx])
    assert len(all_idx) == len(set(all_idx.tolist())) == 400
    meta = read_meta(cfg["paths"]["data_file"])
    if mode == "stratified":                       # every species with >= 3 crowns is in every split
        for ix in (sp.train_idx, sp.val_idx, sp.test_idx):
            assert set(meta["labels"][ix]) == set(meta["labels"])
    else:
        sites = lambda ix: {meta["site"][i] for i in ix}  # noqa: E731
        assert not sites(sp.train_idx) & sites(sp.test_idx) and not sites(sp.train_idx) & sites(sp.val_idx)


def test_demo_dataset_items(rs_cfg) -> None:
    tr, va, te, _ = build_datasets(rs_cfg)
    assert tr.num_classes == 20 and tr.stats["class_names"][0] == "Red maple"
    for ds in (tr, va):
        it = ds[0]
        assert it["rgb"].shape == (3, 32, 32) and it["hsi"].shape == (369, 8, 8) and it["lidar"].shape == (512, 4)
        assert all(torch.isfinite(it[k]).all() for k in ("rgb", "hsi", "lidar")) and it["modality_mask"].sum() == 3
    a, b = va[1], va[1]
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_predictor_on_file_crowns_and_missing_sensors(rs_ckpt, rs_cfg) -> None:
    p = TreePredictor(rs_ckpt, device="cpu")
    tr, _, _, _ = build_datasets(rs_cfg)
    rgb, hsi, chm = tr.raw(int(tr.indices[0]))
    out = p.predict([(rgb, hsi, chm), (rgb, None, None), (None, hsi, chm)])
    assert out[0].sensors_used == ["rgb", "hsi", "lidar"] and out[1].sensors_used == ["rgb"]
    assert out[2].sensors_used == ["hsi", "lidar"] and out[1].weight["hsi"] == 0.0
    for o in out:
        assert len(o.top_species) == 5 and abs(sum(pr for *_, pr in o.top_species)) <= 1.0001
        assert o.top_species[0][2] >= o.top_species[-1][2] and o.height_m >= 0 and o.crown_diameter_m >= 0
        assert o.species in p.names
    dropped = p.predict([(rgb, hsi, chm)], drop=["lidar"])[0]
    assert dropped.sensors_used == ["rgb", "hsi"]


def test_user_file_loaders(tmp_path) -> None:
    from PIL import Image
    Image.fromarray(np.full((100, 80, 3), 120, np.uint8)).save(tmp_path / "c.png")
    assert load_user_rgb(tmp_path / "c.png", (3, 32, 32)).shape == (3, 32, 32)
    np.save(tmp_path / "s.npy", np.full(369, 2500.0))              # single spectrum scaled x10000
    h = load_user_hsi(tmp_path / "s.npy", (369, 8, 8))
    assert h.shape == (369, 8, 8) and np.allclose(h, 0.25)
    chm = np.zeros((64, 64)); chm[:32, 32:] = 12.0                   # top-right quadrant tall
    np.save(tmp_path / "l.npy", chm[None])
    small = load_user_lidar(tmp_path / "l.npy", (1, 32, 32))
    assert small.shape == (32, 32) and small[:16, 16:].min() == 12.0 and small[16:, :16].max() == 0.0
    np.save(tmp_path / "bad.npy", np.ones((100, 8, 8)))
    with pytest.raises(ValueError):
        load_user_hsi(tmp_path / "bad.npy", (369, 8, 8))


def test_predict_script_end_to_end(rs_ckpt, rs_cfg, tmp_path) -> None:
    script = str(PROJECT_ROOT / "scripts" / "predict.py")
    base = [sys.executable, script, "--checkpoint", str(rs_ckpt), "--data", rs_cfg["paths"]["data_file"],
            "--no-open", "--device", "cpu"]
    r = subprocess.run(base + ["--random", "3", "--out", str(tmp_path / "p1")], capture_output=True, text=True,
                       encoding="utf-8")
    assert r.returncode == 0, r.stderr
    assert "Summary over 3 crowns" in r.stdout
    assert (tmp_path / "p1" / "report.html").stat().st_size > 10000
    assert len(list((tmp_path / "p1" / "cards").glob("*.png"))) == 3
    r = subprocess.run(base + ["--export-examples", str(tmp_path / "ex")], capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr
    t = json.loads((tmp_path / "ex" / "crown_1_truth.json").read_text(encoding="utf-8"))
    r = subprocess.run(base + ["--rgb", str(tmp_path / "ex" / "crown_1_rgb.png"), "--lidar",
                               str(tmp_path / "ex" / "crown_1_lidar.npy"), "--out", str(tmp_path / "p2")],
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr
    assert t["species"] in r.stdout and "HSI missing" in r.stdout
