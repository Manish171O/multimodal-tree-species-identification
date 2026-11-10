"""Training loop: modality dropout, focal loss, class-balanced sampling, AMP, checkpoints, JSONL logs."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import build_datasets
from src.evaluation.metrics import basic_metrics
from src.evaluation.predict import predict
from src.losses import TotalLoss, class_balanced_weights
from src.models.model import MultimodalTreeNet, build_model
from src.training.modality_dropout import sample_modality_keep
from src.training.sampler import class_balanced_sampler
from src.utils.common import get_device, get_logger, set_seed


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Move every tensor in a batch dict to ``device``."""
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def warmup_cosine(step: int, total: int, warmup: int) -> float:
    """LR multiplier: linear warm-up for ``warmup`` steps, then cosine decay to 1 % at ``total``."""
    if step < warmup:
        return (step + 1) / max(1, warmup)
    prog = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * prog))


class Trainer:
    """Builds data, model, loss and optimizer from a config and runs training with validation."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        t = cfg["train"]
        set_seed(cfg["seed"], deterministic=True)
        self.out_dir = Path(cfg["paths"]["output_dir"])
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log = get_logger("trainer", str(self.out_dir / "train.log"))
        self.device = get_device(t.get("device", "auto"))
        self.amp = bool(t.get("amp", False)) and self.device.type == "cuda"

        # ---- data
        self.train_ds, self.val_ds, self.test_ds, self.split = build_datasets(cfg)
        d = cfg["data"]
        kw = dict(batch_size=d["batch_size"], num_workers=d["num_workers"], pin_memory=self.device.type == "cuda")
        sampler = None
        if t.get("sampler", "class_balanced") == "class_balanced":
            sampler = class_balanced_sampler(self.train_ds.labels, float(t.get("sampler_power", 0.5)),
                                             seed=cfg["seed"])
        self.train_loader = DataLoader(self.train_ds, sampler=sampler, shuffle=sampler is None,
                                       generator=torch.Generator().manual_seed(cfg["seed"]), **kw)
        self.val_loader = DataLoader(self.val_ds, shuffle=False, **kw)

        # ---- model / loss / optim
        self.model: MultimodalTreeNet = build_model(cfg, self.train_ds).to(self.device)
        lc = cfg["loss"]
        counts = np.bincount(self.train_ds.labels, minlength=self.train_ds.num_classes)
        alpha = class_balanced_weights(counts, float(lc.get("cb_beta", 0.999))) if lc.get("class_balanced_alpha") else None
        self.criterion = TotalLoss(lc, self.model.modalities, alpha).to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=float(t["lr"]), weight_decay=float(t["weight_decay"]))
        steps = int(t["epochs"]) * max(1, len(self.train_loader))
        warm = int(t.get("warmup_epochs", 1)) * max(1, len(self.train_loader))
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda s: warmup_cosine(s, steps, warm))
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.md_gen = torch.Generator().manual_seed(cfg["seed"] + 1)
        self.start_epoch, self.best, self.history = 0, -1.0, []

        (self.out_dir / "split.json").write_text(json.dumps(self.split.as_dict(), indent=1), encoding="utf-8")
        (self.out_dir / "config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        self.log.info(f"device={self.device} amp={self.amp} train={len(self.train_ds)} val={len(self.val_ds)} "
                      f"test={len(self.test_ds)} classes={self.train_ds.num_classes} "
                      f"params={sum(p.numel() for p in self.model.parameters()):,}")

    # ----------------------------------------------------------------- epochs
    def train_epoch(self, epoch: int) -> dict[str, float]:
        """One pass over the training loader. Returns mean loss components and accuracy."""
        self.model.train()
        self.train_ds.epoch = epoch
        md = self.cfg["train"]["modality_dropout"]
        sums: dict[str, float] = {}
        n, correct = 0, 0
        for batch in self.train_loader:
            batch = to_device(batch, self.device)
            keep = None
            if md.get("enabled", True):
                keep = sample_modality_keep(batch["modality_mask"], float(md["p_apply"]), tuple(md["p_drop"]),
                                            self.md_gen).to(self.device)                     # (B, 3)
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.amp):
                out = self.model(batch, keep=keep)
                losses = self.criterion(out, batch, self.model.target_mean, self.model.target_std, epoch)
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}: { {k: float(v) for k, v in losses.items()} }")
            self.opt.zero_grad(set_to_none=True)
            self.scaler.scale(losses["total"]).backward()
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg["train"]["grad_clip"]))
            self.scaler.step(self.opt)
            self.scaler.update()
            self.sched.step()
            bs = batch["label"].shape[0]
            n += bs
            correct += int((out["logits"].argmax(1) == batch["label"]).sum())
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + float(v.detach()) * bs
        res = {f"train_{k}": v / max(n, 1) for k, v in sums.items()}
        res["train_acc"] = correct / max(n, 1)
        res["lr"] = self.opt.param_groups[0]["lr"]
        return res

    def predict(self, loader: DataLoader, keep: tuple[int, int, int] | None = None) -> dict[str, np.ndarray]:
        """Run the model over ``loader``; ``keep`` optionally drops sensors for every sample."""
        return predict(self.model, loader, self.device, keep, self.amp)

    def validate(self) -> dict[str, float]:
        """Validation metrics (all sensors present as recorded)."""
        p = self.predict(self.val_loader)
        m = basic_metrics(p["label"], p["pred"], self.train_ds.num_classes)
        res = {f"val_{k}": v for k, v in m.items()}
        res["val_height_mae"] = float(np.abs(p["structure"][:, 0] - p["struct_true"][:, 0]).mean())
        present = p["mask"] > 0
        for i, name in enumerate(self.model.modalities):
            res[f"val_trust_{name}"] = float(p["trust"][present[:, i], i].mean()) if present[:, i].any() else 0.0
        return res

    # ------------------------------------------------------------ checkpoints
    def save(self, name: str, epoch: int, metrics: dict[str, float]) -> Path:
        """Save model + optimizer + scheduler + RNG states for exact resume."""
        path = self.out_dir / name
        torch.save({"epoch": epoch, "model": self.model.state_dict(), "opt": self.opt.state_dict(),
                    "sched": self.sched.state_dict(), "scaler": self.scaler.state_dict(), "best": self.best,
                    "metrics": metrics, "cfg": self.cfg, "norm_stats": self.train_ds.stats,
                    "num_classes": self.train_ds.num_classes, "history": self.history,
                    "md_gen": self.md_gen.get_state(), "torch_rng": torch.get_rng_state()}, path)
        return path

    def resume(self, path: str | Path) -> None:
        """Restore training state from a checkpoint written by :meth:`save`."""
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["opt"])
        self.sched.load_state_dict(ck["sched"])
        self.scaler.load_state_dict(ck["scaler"])
        self.md_gen.set_state(ck["md_gen"])
        torch.set_rng_state(ck["torch_rng"])
        self.start_epoch, self.best, self.history = ck["epoch"] + 1, ck["best"], ck["history"]
        self.log.info(f"resumed from {path} at epoch {self.start_epoch}")

    # -------------------------------------------------------------------- fit
    def fit(self) -> list[dict[str, float]]:
        """Train for ``train.epochs`` with early stopping on ``train.select_metric``. Returns history."""
        t = self.cfg["train"]
        metric, patience = t.get("select_metric", "val_macro_f1"), int(t.get("early_stop_patience", 10))
        bad = 0
        log_path = self.out_dir / "metrics.jsonl"
        for epoch in range(self.start_epoch, int(t["epochs"])):
            t0 = time.time()
            rec = {"epoch": epoch, **self.train_epoch(epoch), **self.validate()}
            rec["epoch_time_s"] = time.time() - t0
            self.history.append(rec)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            improved = rec[metric] > self.best
            if improved:
                self.best, bad = rec[metric], 0
                self.save("best.pt", epoch, rec)
            else:
                bad += 1
            self.save("last.pt", epoch, rec)
            self.log.info(
                f"ep {epoch:3d} | loss {rec['train_total']:.3f} (cls {rec['train_cls']:.3f} con {rec['train_contrastive']:.3f} "
                f"aux {rec['train_aux']:.3f} evi {rec['train_evidential']:.3f}) | train_acc {rec['train_acc']:.3f} | "
                f"val_acc {rec['val_accuracy']:.3f} val_mF1 {rec['val_macro_f1']:.3f} | trust r/h/l "
                f"{rec['val_trust_rgb']:.2f}/{rec['val_trust_hsi']:.2f}/{rec['val_trust_lidar']:.2f} | "
                f"{rec['epoch_time_s']:.1f}s{' *' if improved else ''}")
            if bad >= patience:
                self.log.info(f"early stop: no {metric} improvement for {patience} epochs")
                break
        return self.history
