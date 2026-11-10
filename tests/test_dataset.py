"""Dataset loading, NaN/invalid handling, normalization and dataloader batching."""
from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch

from src.data.dataset import build_dataloaders, build_datasets, clean_hsi, clean_lidar
from src.data.inspect import describe_h5, validate_schema


def test_inspect_reads_real_keys(tiny_h5) -> None:
    names = {r["name"] for r in describe_h5(tiny_h5)}
    for k in ("rgb", "hsi", "lidar", "labels/species_id", "meta/site", "hsi_wavelengths"):
        assert k in names
    shapes = validate_schema(tiny_h5)
    assert shapes["hsi"][1] == 369 and shapes["rgb"][1] == 3 and shapes["lidar"][2] == 4


def test_validate_schema_rejects_bad_file(tmp_path) -> None:
    p = tmp_path / "bad.h5"
    with h5py.File(p, "w") as f:
        f["rgb"] = np.zeros((2, 3, 4, 4), np.uint8)
    with pytest.raises(KeyError):
        validate_schema(p)


def test_clean_hsi_handles_nodata_nan_saturation() -> None:
    raw = np.full((5, 3, 3), 2000.0, np.float32)
    raw[:, 0, 0] = -9999            # nodata pixel
    raw[2, 1, 1] = np.nan           # NaN in one band -> whole pixel invalid
    raw[0, 2, 2] = 30000            # saturated
    raw[1, 0, 1] = -50              # small negative -> clipped to 0, still valid
    x, vpx = clean_hsi(raw, 10000.0, -9999)
    assert vpx.sum() == 9 - 3
    assert not vpx[0, 0] and not vpx[1, 1] and not vpx[2, 2]
    assert np.isnan(x[:, ~vpx]).all() and np.isfinite(x[:, vpx]).all()
    assert x[1, 0, 1] == 0.0 and np.isclose(x[0, 0, 1], 0.2)


def test_clean_lidar_drops_invalid_rows() -> None:
    raw = np.array([[0, 0, 1, 0.5], [np.nan, 0, 1, 0.5], [0, 0, np.inf, 0.5], [1, 1, 2, np.nan]], np.float32)
    out = clean_lidar(raw)
    assert out.shape == (2, 4)       # NaN intensity row kept (filled later), bad xyz rows dropped


def test_item_shapes_dtypes_and_finite(cfg) -> None:
    tr, va, te, _ = build_datasets(cfg)
    for ds in (tr, va, te):
        assert len(ds) > 0
        for j in range(min(len(ds), 40)):
            it = ds[j]
            assert it["rgb"].shape == (3, 64, 64) and it["rgb"].dtype == torch.float32
            assert it["hsi"].shape == (369, 12, 12)
            assert it["lidar"].shape == (cfg["data"]["lidar_num_points"], 4)
            assert it["lidar_mask"].dtype == torch.bool and it["modality_mask"].shape == (3,)
            assert it["label"].dtype == torch.int64
            for k in ("rgb", "hsi", "lidar"):
                assert torch.isfinite(it[k]).all(), f"non-finite values in {k}"
            assert (it["lidar"][~it["lidar_mask"]] == 0).all()


def test_missing_modalities_are_zero_and_masked(cfg, tiny_h5) -> None:
    tr, va, te, split = build_datasets(cfg)
    with h5py.File(tiny_h5, "r") as f:
        truth = f["synthetic_truth/missing"][:]
    seen = np.zeros(3, bool)
    for ds in (tr, va, te):
        for j, i in enumerate(ds.indices):
            for m, key in enumerate(("rgb", "hsi", "lidar")):
                if truth[i, m]:
                    it = ds[j]
                    assert it["modality_mask"][m] == 0
                    assert (it[key] == 0).all()
                    seen[m] = True
    assert seen.all(), "fixture should contain a missing case for every modality"


def test_normalization_fit_on_train_only(cfg) -> None:
    tr, _, _, split = build_datasets(cfg)
    assert tr.stats["train_sites"] == split.train_sites
    assert len(tr.stats["hsi_mean"]) == 369
    rgb = torch.stack([tr[j]["rgb"] for j in range(len(tr)) if tr[j]["modality_mask"][0] == 1])
    assert rgb.mean(dim=(0, 2, 3)).abs().max() < 0.25       # roughly z-scored on train
    assert (rgb.std(dim=(0, 2, 3)) - 1).abs().max() < 0.3


def test_eval_items_deterministic(cfg) -> None:
    _, va, _, _ = build_datasets(cfg)
    a, b = va[0], va[0]
    for k in a:
        assert torch.equal(a[k], b[k])


def test_dataloader_batches(cfg) -> None:
    trl, val, tel, _ = build_dataloaders(cfg)
    batch = next(iter(trl))
    bs = cfg["data"]["batch_size"]
    assert batch["rgb"].shape == (bs, 3, 64, 64)
    assert batch["hsi"].shape == (bs, 369, 12, 12)
    assert batch["lidar"].shape == (bs, cfg["data"]["lidar_num_points"], 4)
    assert batch["modality_mask"].shape == (bs, 3)
    n = sum(len(dl.dataset) for dl in (trl, val, tel))
    with h5py.File(cfg["paths"]["data_file"], "r") as f:
        assert n == len(f["rgb"])


def test_brightness_normalize_removes_gain() -> None:
    from src.data.dataset import brightness_normalize
    rng = np.random.default_rng(0)
    x = rng.uniform(0.05, 0.6, (20, 4, 4)).astype(np.float32)
    vpx = np.ones((4, 4), bool)
    vpx[0, 0] = False
    x[:, 0, 0] = np.nan
    a, b = brightness_normalize(x, vpx), brightness_normalize(1.7 * x, vpx)
    assert np.allclose(a[:, vpx], b[:, vpx], atol=1e-5) and np.isnan(a[:, 0, 0]).all()
    assert np.allclose(np.nanmean(a[:, vpx], axis=0), 1.0, atol=1e-5)


def test_spectral_augment_shape_and_nan() -> None:
    from src.data.dataset import spectral_augment
    x = np.full((50, 3, 3), 0.3, np.float32)
    x[:, 1, 1] = np.nan
    y = spectral_augment(x, np.random.default_rng(1), {"gain_sigma": 0.2, "tilt_sigma": 0.1})
    assert y.shape == x.shape and np.isnan(y[:, 1, 1]).all() and not np.allclose(y[:, 0, 0], 0.3)


def test_robustness_options_give_finite_items(cfg) -> None:
    from src.utils.common import deep_merge
    c = deep_merge(cfg, {"data": {"hsi_brightness_norm": True, "hsi_aug": {"enabled": True},
                                  "rgb_aug": {"enabled": True}}})
    tr, va, _, _ = build_datasets(c)
    assert tr.stats["hsi_brightness_norm"] is True
    for ds in (tr, va):
        for j in range(10):
            it = ds[j]
            assert torch.isfinite(it["hsi"]).all() and torch.isfinite(it["rgb"]).all()
