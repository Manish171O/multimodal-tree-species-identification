"""Smoke test: 2 epochs on 64 training samples must finish in < 2 minutes on CPU, and be reproducible."""
from __future__ import annotations

import json
import math
import time

import torch

from src.training.trainer import Trainer
from src.utils.common import deep_merge

SMOKE = {"data": {"max_train_samples": 64, "max_eval_samples": 32, "batch_size": 16},
         "train": {"epochs": 2, "device": "cpu", "warmup_epochs": 1}}


def _run(cfg: dict) -> tuple[list[dict], float, Trainer]:
    t0 = time.time()
    tr = Trainer(deep_merge(cfg, SMOKE))
    hist = tr.fit()
    return hist, time.time() - t0, tr


def test_smoke_train_two_epochs_cpu(cfg) -> None:
    hist, elapsed, tr = _run(cfg)
    assert elapsed < 120, f"smoke run took {elapsed:.1f}s"
    assert len(tr.train_ds) == 64 and len(hist) == 2
    for rec in hist:
        for k in ("train_total", "train_cls", "train_contrastive", "train_aux", "train_evidential", "val_macro_f1"):
            assert math.isfinite(rec[k]), k
    out = tr.out_dir
    assert (out / "best.pt").exists() and (out / "last.pt").exists() and (out / "split.json").exists()
    lines = (out / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2 and json.loads(lines[-1])["epoch"] == 1
    ck = torch.load(out / "last.pt", weights_only=False)
    assert ck["epoch"] == 1 and "model" in ck and ck["num_classes"] == tr.train_ds.num_classes
    print(f"\nsmoke: {elapsed:.1f}s, epoch losses {[round(r['train_total'], 3) for r in hist]}")


def test_training_is_reproducible(cfg, tmp_path) -> None:
    c1 = deep_merge(cfg, {"paths": {"output_dir": str(tmp_path / "a")}})
    c2 = deep_merge(cfg, {"paths": {"output_dir": str(tmp_path / "b")}})
    h1, _, _ = _run(c1)
    h2, _, _ = _run(c2)
    for a, b in zip(h1, h2):
        assert a["train_total"] == b["train_total"] and a["val_macro_f1"] == b["val_macro_f1"]


def test_resume_continues_from_checkpoint(cfg) -> None:
    _, _, tr = _run(cfg)
    tr2 = Trainer(deep_merge(cfg, deep_merge(SMOKE, {"train": {"epochs": 3}})))
    tr2.resume(tr.out_dir / "last.pt")
    hist = tr2.fit()
    assert tr2.start_epoch == 2 and hist[-1]["epoch"] == 2 and len(hist) == 3
