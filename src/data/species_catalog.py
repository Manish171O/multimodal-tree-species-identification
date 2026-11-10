"""Catalog of 20 real North American tree species and 10 NEON field sites for the demo generator.

Species names, families, leaf types and site occurrences are real (site lists are approximate
ranges of where each species commonly occurs at or near that NEON site). Every *numeric* trait
below (heights, crown ratios, reflectance shape parameters, colours) is a plausible, hand-set
value chosen so species differ in realistic directions (e.g. conifers are darker and have lower
NIR than broadleaves; tulip poplar is tall; turkey oak is short). They are NOT measured values.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Species:
    code: str              # NEON-style taxon code
    scientific: str
    common: str
    family: str
    leaf: str              # "broadleaf" | "needleleaf"
    height: float          # typical mature canopy height, m
    crown_ratio: float     # crown diameter / height
    shape: str             # crown profile in the height raster: "dome" | "flat" | "cone" | "column"
    green: float           # visible green-peak reflectance
    red_edge: float        # red-edge inflection, nm
    nir: float             # NIR plateau reflectance
    swir: float            # SWIR (~1650 nm) reflectance
    water: float           # depth of leaf-water absorption features (0-1)
    rgb_tint: tuple[float, float, float]  # multiplicative colour cast of the crown in RGB
    texture: float         # crown roughness in RGB / height raster (0-1)
    abundance: float       # relative abundance weight (long-tailed)
    sites: tuple[str, ...]


SPECIES: tuple[Species, ...] = (
    # ---------------- broadleaf (deciduous) ----------------
    Species("ACRU", "Acer rubrum", "Red maple", "Sapindaceae", "broadleaf", 21, 0.45, "dome",
            0.110, 716, 0.48, 0.24, 0.55, (1.08, 0.98, 0.92), 0.45, 10.0,
            ("HARV", "BART", "SCBI", "SERC", "MLBS", "OSBS", "UNDE")),
    Species("ACSA3", "Acer saccharum", "Sugar maple", "Sapindaceae", "broadleaf", 26, 0.50, "dome",
            0.120, 720, 0.52, 0.25, 0.55, (1.00, 1.04, 0.92), 0.40, 5.0, ("HARV", "BART", "UNDE")),
    Species("QURU", "Quercus rubra", "Northern red oak", "Fagaceae", "broadleaf", 25, 0.55, "flat",
            0.095, 718, 0.45, 0.22, 0.60, (0.98, 1.00, 0.90), 0.55, 6.0,
            ("HARV", "SCBI", "MLBS", "UNDE", "BART")),
    Species("QUAL", "Quercus alba", "White oak", "Fagaceae", "broadleaf", 24, 0.60, "flat",
            0.100, 719, 0.47, 0.23, 0.58, (1.00, 0.98, 0.95), 0.55, 4.0, ("SCBI", "MLBS", "SERC")),
    Species("QULA2", "Quercus laevis", "Turkey oak", "Fagaceae", "broadleaf", 9, 0.70, "flat",
            0.090, 712, 0.38, 0.24, 0.45, (1.02, 0.96, 0.88), 0.65, 3.0, ("OSBS",)),
    Species("FAGR", "Fagus grandifolia", "American beech", "Fagaceae", "broadleaf", 22, 0.55, "dome",
            0.125, 722, 0.55, 0.26, 0.55, (0.95, 1.05, 0.95), 0.30, 4.5, ("HARV", "BART", "SCBI", "SERC")),
    Species("BEAL2", "Betula alleghaniensis", "Yellow birch", "Betulaceae", "broadleaf", 20, 0.45, "dome",
            0.130, 719, 0.53, 0.27, 0.50, (1.03, 1.06, 0.88), 0.40, 3.0, ("HARV", "BART", "UNDE")),
    Species("LITU", "Liriodendron tulipifera", "Tulip poplar", "Magnoliaceae", "broadleaf", 33, 0.35, "column",
            0.135, 724, 0.58, 0.27, 0.52, (0.98, 1.08, 0.90), 0.35, 4.0, ("SCBI", "SERC", "MLBS")),
    Species("LIST2", "Liquidambar styraciflua", "Sweetgum", "Altingiaceae", "broadleaf", 24, 0.45, "dome",
            0.115, 717, 0.50, 0.25, 0.55, (1.00, 1.02, 0.88), 0.45, 3.0, ("SERC", "TALL", "OSBS")),
    Species("NYSY", "Nyssa sylvatica", "Black gum", "Nyssaceae", "broadleaf", 18, 0.45, "dome",
            0.100, 715, 0.46, 0.24, 0.58, (1.05, 0.96, 0.92), 0.40, 2.0, ("SCBI", "SERC", "MLBS", "OSBS")),
    Species("POTR5", "Populus tremuloides", "Quaking aspen", "Salicaceae", "broadleaf", 17, 0.35, "column",
            0.140, 721, 0.57, 0.28, 0.45, (1.00, 1.10, 0.85), 0.25, 2.5, ("UNDE", "NIWO")),
    # ---------------- needleleaf (conifers) ----------------
    Species("PIST", "Pinus strobus", "Eastern white pine", "Pinaceae", "needleleaf", 28, 0.40, "cone",
            0.075, 712, 0.34, 0.17, 0.70, (0.92, 1.00, 1.02), 0.60, 5.0, ("HARV", "BART", "UNDE")),
    Species("TSCA", "Tsuga canadensis", "Eastern hemlock", "Pinaceae", "needleleaf", 24, 0.35, "cone",
            0.065, 710, 0.30, 0.15, 0.72, (0.88, 0.98, 1.00), 0.50, 5.0, ("HARV", "BART", "MLBS")),
    Species("ABBA", "Abies balsamea", "Balsam fir", "Pinaceae", "needleleaf", 15, 0.30, "cone",
            0.060, 708, 0.29, 0.14, 0.75, (0.85, 0.98, 1.02), 0.35, 2.5, ("BART", "UNDE")),
    Species("PIPA2", "Pinus palustris", "Longleaf pine", "Pinaceae", "needleleaf", 25, 0.35, "flat",
            0.085, 710, 0.32, 0.19, 0.62, (0.96, 1.00, 0.95), 0.75, 6.0, ("OSBS", "TALL")),
    Species("PITA", "Pinus taeda", "Loblolly pine", "Pinaceae", "needleleaf", 27, 0.35, "dome",
            0.080, 711, 0.35, 0.18, 0.65, (0.94, 1.00, 0.97), 0.65, 6.0, ("TALL", "SERC", "OSBS")),
    Species("PIEN", "Picea engelmannii", "Engelmann spruce", "Pinaceae", "needleleaf", 22, 0.25, "column",
            0.058, 707, 0.27, 0.13, 0.75, (0.86, 0.96, 1.05), 0.40, 4.0, ("NIWO",)),
    Species("PICOL", "Pinus contorta", "Lodgepole pine", "Pinaceae", "needleleaf", 18, 0.25, "column",
            0.075, 709, 0.31, 0.17, 0.66, (0.95, 1.00, 0.95), 0.55, 4.0, ("NIWO",)),
    Species("PSMEM", "Pseudotsuga menziesii", "Douglas-fir", "Pinaceae", "needleleaf", 45, 0.25, "cone",
            0.068, 710, 0.32, 0.15, 0.72, (0.88, 0.98, 1.00), 0.55, 5.0, ("WREF",)),
    Species("TSHE", "Tsuga heterophylla", "Western hemlock", "Pinaceae", "needleleaf", 38, 0.25, "cone",
            0.064, 709, 0.30, 0.15, 0.74, (0.87, 0.97, 1.01), 0.50, 4.0, ("WREF",)),
)

# Real NEON terrestrial sites (codes, names, states).
SITES: dict[str, str] = {
    "HARV": "Harvard Forest, MA",
    "BART": "Bartlett Experimental Forest, NH",
    "SCBI": "Smithsonian Conservation Biology Institute, VA",
    "SERC": "Smithsonian Environmental Research Center, MD",
    "MLBS": "Mountain Lake Biological Station, VA",
    "OSBS": "Ordway-Swisher Biological Station, FL",
    "TALL": "Talladega National Forest, AL",
    "UNDE": "University of Notre Dame Environmental Research Center, MI",
    "NIWO": "Niwot Ridge, CO",
    "WREF": "Wind River Experimental Forest, WA",
}
