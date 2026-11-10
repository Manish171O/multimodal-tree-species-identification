"""Metric correctness and the end-to-end evaluation suite on a tiny trained checkpoint."""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.evaluation.evaluate import SCENARIOS, evaluate, summary_table
from src.evaluation.metrics import (basic_metrics, confusion_matrix, expected_calibration_error,
                                    per_class_report, top_k_accuracy)
from src.training.trainer import Trainer
from src.utils.common import deep_merge


def test_basic_metrics_hand_computed() -> None:
    y = np.array([0, 0, 0, 1, 1, 2])
    p = np.array([0, 0, 1, 1, 2, 2])
    m = basic_metrics(y, p, 4)                        # class 3 absent -> excluded from macro averages
    # per-class P/R/F1: c0 (1, 2/3, .8), c1 (.5, .5, .5), c2 (.5, 1, 2/3)
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["macro_precision"] == pytest.approx((1 + 0.5 + 0.5) / 3)
    assert m["macro_recall"] == pytest.approx((2 / 3 + 0.5 + 1) / 3)
    assert m["macro_f1"] == pytest.approx((0.8 + 0.5 + 2 / 3) / 3)
    assert m["weighted_f1"] == pytest.approx((3 * 0.8 + 2 * 0.5 + 1 * 2 / 3) / 6)
    cm = confusion_matrix(y, p, 4)
    assert cm.sum() == 6 and cm[0, 1] == 1 and cm[1, 2] == 1


def test_ece_zero_when_calibrated_and_known_when_overconfident() -> None:
    rng = np.random.default_rng(0)
    n, c = 20000, 4
    conf = rng.uniform(0.3, 1.0, n)
    pred = rng.integers(0, c, n)
    correct = rng.random(n) < conf                     # accuracy == confidence -> calibrated
    y = np.where(correct, pred, (pred + 1) % c)
    prob = np.full((n, c), 0.0)
    prob[np.arange(n), pred] = conf
    prob[np.arange(n), (pred + 1) % c] = 1 - conf      # keeps argmax = pred for conf > 0.5 ...
    keep = conf > 0.5                                  # ... so evaluate only those rows
    ece, bins = expected_calibration_error(prob[keep], y[keep], 15)
    assert ece < 0.02 and bins["count"].sum() == keep.sum()
    over = np.zeros((100, 2))
    over[:, 0] = 1.0                                   # always 100% confident, 50% right
    ece2, _ = expected_calibration_error(over, np.array([0, 1] * 50))
    assert ece2 == pytest.approx(0.5)


def test_top_k() -> None:
    prob = np.array([[0.5, 0.3, 0.2], [0.1, 0.2, 0.7]])
    assert top_k_accuracy(prob, np.array([2, 0]), 1) == 0.0
    assert top_k_accuracy(prob, np.array([1, 1]), 2) == 1.0


def test_per_class_report_rows() -> None:
    rows = per_class_report(np.array([0, 1, 1]), np.array([0, 1, 2]), 5, train_counts=np.array([3, 1, 0, 0, 0]))
    assert [r["class_id"] for r in rows] == [0, 1, 2]
    assert rows[1]["recall"] == 0.5 and rows[2]["train_count"] == 0


@pytest.fixture()
def tiny_checkpoint(cfg):
    c = deep_merge(cfg, {"data": {"max_train_samples": 64, "batch_size": 16},
                         "train": {"epochs": 1, "device": "cpu", "warmup_epochs": 0}})
    tr = Trainer(c)
    tr.fit()
    return tr.out_dir / "best.pt"


def test_evaluate_end_to_end(tiny_checkpoint, tmp_path) -> None:
    out = tmp_path / "eval"
    res = evaluate(tiny_checkpoint, out, "test", device="cpu")
    assert set(res["scenarios"]) == set(SCENARIOS)
    for name, r in res["scenarios"].items():
        keep = np.array(r["keep"])
        for i, m in enumerate(("rgb", "hsi", "lidar")):
            if keep[i] == 0:
                assert r["mean_fusion_weight"][m] == 0.0, f"{name}: dropped {m} still weighted"
        assert 0 <= r["all"]["accuracy"] <= 1 and 0 <= r["all"]["ece"] <= 1
        assert r["seen_species"]["n"] <= r["all"]["n"] == res["n_samples"]
    assert res["scenarios"]["all_sensors"]["all"]["accuracy_drop"] == 0.0
    for f in ("results.json", "per_species.csv", "per_site.csv", "confusion_matrix.csv"):
        assert (out / f).exists()
    for f in ("missing_modality", "reliability", "confusion_matrix", "per_species_f1", "trust_weights"):
        assert (out / "plots" / f"{f}.png").stat().st_size > 1000
    assert json.loads((out / "results.json").read_text(encoding="utf-8"))["synthetic_data"] is True
    assert "trust_vs_degradation_synthetic_truth" in res and "height_m" in res["structure"]
    assert summary_table(res).count("\n") == len(SCENARIOS) + 1


def test_report_from_evaluations(tiny_checkpoint, tmp_path) -> None:
    from src.evaluation.report import write_report
    run = {"val": evaluate(tiny_checkpoint, tmp_path / "v", "val", device="cpu", make_plots=False),
           "test": evaluate(tiny_checkpoint, tmp_path / "t", "test", device="cpu", make_plots=False),
           "overrides": {}, "train_time": "1s"}
    summary = {"selected_variant": "baseline", "quick": True,
               "runs": {"baseline": run, "baseline__no_contrastive": run}}
    md = write_report(summary, tmp_path)
    assert "SYNTHETIC" in md and "no_contrastive" in md and "(+0.000)" in md
    assert (tmp_path / "results.md").exists()
    assert (tmp_path / "plots" / "comparison.png").exists() and (tmp_path / "plots" / "missing_sensors.png").exists()
