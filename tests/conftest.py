"""Shared pytest fixtures: a small SYNTHETIC HDF5 file and a matching config."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.synthetic import SyntheticConfig, generate_synthetic_dataset  # noqa: E402
from src.utils.common import load_config  # noqa: E402


@pytest.fixture(scope="session")
def tiny_h5(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """300 synthetic crowns, 20 species, 8 sites, real-sized tensors; higher missing rates for coverage."""
    path = tmp_path_factory.mktemp("data") / "tiny.h5"
    scfg = SyntheticConfig(num_samples=300, num_species=20, num_sites=8,
                           p_missing_rgb=0.08, p_missing_hsi=0.08, p_missing_lidar=0.08)
    return generate_synthetic_dataset(path, scfg, seed=123)


@pytest.fixture()
def cfg(tiny_h5: Path, tmp_path: Path) -> dict:
    """Default config redirected to the tiny file and a per-test output dir."""
    return load_config(ROOT / "configs" / "default.yaml", overrides={
        "paths": {"data_file": str(tiny_h5), "output_dir": str(tmp_path / "out"),
                  "norm_stats": str(tmp_path / "out" / "norm_stats.json")},
        "data": {"batch_size": 8},
    })
