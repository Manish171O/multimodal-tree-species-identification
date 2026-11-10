"""Shared utilities: config loading, seeding, device selection, logging."""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


def load_config(path: str | os.PathLike, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load a YAML config and optionally deep-merge ``overrides`` into it.

    Relative entries under ``paths`` are resolved against the project root.
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    for key, value in cfg.get("paths", {}).items():
        p = Path(value)
        cfg["paths"][key] = str(p if p.is_absolute() else PROJECT_ROOT / p)
    return cfg


def deep_merge(base: dict[str, Any], upd: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``upd`` into a copy of ``base``."""
    out = dict(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Fix Python, NumPy and PyTorch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(prefer: str = "auto") -> torch.device:
    """Return CUDA if available (or requested), else CPU."""
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_logger(name: str = "treespecies", log_file: str | None = None) -> logging.Logger:
    """Console (+ optional file) logger with a compact format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(fh)
    return logger
