"""Real-species demo generator: schema of demo_tree_crowns.h5, real names, biogeography, reproducibility."""
from __future__ import annotations

import h5py
import numpy as np
import pytest

from src.data.real_species import generate_real_species_dataset
from src.data.species_catalog import SITES, SPECIES

# Layout of the reference file demo_tree_crowns.h5: key -> (dtype, per-sample shape)
REFERENCE = {"rgb": ("uint8", (3, 32, 32)), "hsi": ("float16", (369, 8, 8)), "lidar": ("float32", (1, 32, 32)),
             "species": ("int64", ()), "site": ("int64", ()), "height": ("float32", ()),
             "crown_diameter": ("float32", ()), "degraded": ("bool", (3,))}


@pytest.fixture(scope="module")
def rs_file(tmp_path_factory):
    return generate_real_species_dataset(tmp_path_factory.mktemp("rs") / "rs.h5", num_samples=400, seed=7)


def test_schema_matches_reference(rs_file) -> None:
    with h5py.File(rs_file, "r") as f:
        assert set(f) == set(REFERENCE) | {"wavelengths"}
        for k, (dt, shp) in REFERENCE.items():
            assert str(f[k].dtype) == dt and f[k].shape == (400,) + shp, k
        assert f["wavelengths"].shape == (369,) and f["wavelengths"].dtype == np.float64
        assert np.isclose(f["wavelengths"][0], 380) and np.isclose(f["wavelengths"][-1], 2510)
        for a in ("note", "site_names", "species_names"):
            assert a in f.attrs
        assert "SYNTHETIC" in f.attrs["note"]


def test_real_names_and_valid_labels(rs_file) -> None:
    with h5py.File(rs_file, "r") as f:
        names, sites = list(f.attrs["species_names"]), list(f.attrs["site_names"])
        assert "Northern red oak" in names and "Douglas-fir" in names
        assert list(f.attrs["species_scientific_names"])[names.index("Northern red oak")] == "Quercus rubra"
        assert set(sites) <= set(SITES)
        sp, st = f["species"][:], f["site"][:]
        assert sp.min() >= 0 and sp.max() < len(names) and st.max() < len(sites)
        assert np.bincount(sp, minlength=len(names)).min() >= 15


def test_species_only_at_their_sites(rs_file) -> None:
    by_name = {s.common: s for s in SPECIES}
    with h5py.File(rs_file, "r") as f:
        names, sites = list(f.attrs["species_names"]), list(f.attrs["site_names"])
        for c, s in zip(f["species"][:], f["site"][:]):
            assert sites[s] in by_name[names[c]].sites


def test_values_finite_and_in_range(rs_file) -> None:
    with h5py.File(rs_file, "r") as f:
        hsi = f["hsi"][:].astype(np.float32)
        assert np.isfinite(hsi).all() and hsi.min() >= 0 and hsi.max() <= 1.2
        assert np.isfinite(f["lidar"][:]).all()
        h, cd = f["height"][:], f["crown_diameter"][:]
        assert (h >= 3).all() and (h <= 60).all() and (cd >= 2.5).all() and (cd <= 14).all()
        d = f["degraded"][:].mean(0)
        assert np.allclose(d, [0.20, 0.15, 0.09], atol=0.06)


def test_height_raster_peaks_at_centre(rs_file) -> None:
    """Height raster peaks at the crown centre and is ~0 at the window corner."""
    with h5py.File(rs_file, "r") as f:
        ok = ~f["degraded"][:, 2]
        lid = f["lidar"][:, 0][ok]
        assert (lid[:, 14:18, 14:18].mean((1, 2)) > lid[:, :3, :3].mean((1, 2)) + 2).mean() > 0.95


def test_reproducible_and_subsets(tmp_path) -> None:
    a = generate_real_species_dataset(tmp_path / "a.h5", 300, seed=3, site_codes=["HARV", "BART", "UNDE"])
    b = generate_real_species_dataset(tmp_path / "b.h5", 300, seed=3, site_codes=["HARV", "BART", "UNDE"])
    with h5py.File(a, "r") as fa, h5py.File(b, "r") as fb:
        assert np.array_equal(fa["rgb"][:], fb["rgb"][:]) and np.array_equal(fa["species"][:], fb["species"][:])
        assert list(fa.attrs["site_names"]) == ["HARV", "BART", "UNDE"]
        assert "Douglas-fir" not in list(fa.attrs["species_names"])     # only lives at WREF
