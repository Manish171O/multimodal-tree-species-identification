# Multimodal Tree Species Identification

**Adaptive quality-aware deep learning that identifies tree species from airborne RGB, hyperspectral and LiDAR data.**

The model looks at each tree crown through three sensors, learns how much to trust each sensor for that tree, and predicts:

- the **species** (with a top-5 ranking and confidence),
- the tree's **height** in metres,
- the **crown diameter** in metres.

It keeps working when a sensor is missing or degraded, and it is evaluated on trees (and forest sites) it never saw during training.

![Example prediction card](results/figures/prediction_card_example.png)

---

## Contents

1. [Highlights](#highlights)
2. [About the data](#about-the-data)
3. [Quick start](#quick-start)
4. [Predicting tree species](#predicting-tree-species)
5. [How the model works](#how-the-model-works)
6. [Results](#results)
7. [Datasets](#datasets)
8. [Training and evaluation](#training-and-evaluation)
9. [Reproducing all experiments](#reproducing-all-experiments)
10. [Project structure](#project-structure)
11. [Tests](#tests)
12. [Limitations and future work](#limitations-and-future-work)

---

## Highlights

- **Three sensors, one model.** A separate neural network encodes each sensor: an RGB camera, a 369-band hyperspectral camera and LiDAR.
- **Per-tree sensor trust.** An evidential "quality estimator" scores how reliable each sensor is for every tree, and the fusion step weights the sensors by that score.
- **Robust to missing sensors.** Sensors are switched off at random during training (modality dropout), so the model still works with one or two sensors.
- **Fair evaluation.** Test trees are never used for training or model selection. A site-held-out split tests whole forests the model has never seen.
- **Ready-made demo.** `scripts/predict.py` produces a visual HTML report with one prediction card per tree.
- **Reproducible.** All settings live in YAML configs, seeds are fixed, and 111 automated tests cover every component.

---

## About the data

No real NEON airborne file was available for this project, so the repository includes two data generators that simulate realistic sensor data:

- **Real-species dataset** (`scripts/make_real_species_data.py`): 2,000 crowns of **20 real North American tree species** at **10 real NEON field sites**. Species names, families and site ranges are real. Heights, crown shapes, colours and spectra are realistic hand-set values, not measurements.
- **Research dataset** (`scripts/make_synthetic_data.py`): 2,000 crowns of **167 species** at 16 sites in a NEON Multi-Modal Tree Crown–like layout, used for the main research experiments and ablations.

Both include the difficulties of real data: rare species, look-alike species, lighting differences between sites, missing sensors, degraded sensors and invalid values.

The reported numbers therefore show that the method works and how its parts interact. To use real data, convert it to one of the documented layouts and set `paths.data_file` in the config. No code changes are needed.

---

## Quick start

### 1. Install

Python 3.12 is recommended.

```bash
git clone https://github.com/Manish171O/multimodal-tree-species-identification.git
cd multimodal-tree-species-identification

python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt
```

For an NVIDIA GPU, install the CUDA build of PyTorch 2.15.0 from [pytorch.org](https://pytorch.org/get-started/locally/). The code uses the GPU automatically.

### 2. Create the data

The data files are not stored in the repository because of their size. The generators are seeded, so they recreate exactly the same data in about 10 seconds each.

```bash
python scripts/make_real_species_data.py
python scripts/make_synthetic_data.py
```

### 3. Check that everything works

```bash
python -m pytest
```

All 111 tests should pass in about a minute on a CPU.

### 4. Run a prediction

```bash
python scripts/predict.py
```

This predicts 8 test trees and opens the report in your browser.

---

## Predicting tree species

`scripts/predict.py` uses the trained model in `results/real_species_model.pt`.

### Common commands

| Command | What it does |
|---|---|
| `python scripts/predict.py` | Predicts 8 random test trees |
| `python scripts/predict.py --random 20` | Predicts 20 random test trees |
| `python scripts/predict.py --per-species 1` | One test tree of each of the 20 species |
| `python scripts/predict.py --random 301` | All 301 test trees |
| `python scripts/predict.py --index 5 17 301` | Specific trees by row number |
| `python scripts/predict.py --drop lidar` | Same, with the LiDAR sensor removed |
| `python scripts/predict.py --per-species 1 --drop hsi lidar` | Every species using RGB only |
| `python scripts/predict.py --export-examples examples` | Saves test trees as image and `.npy` files |

Use `--out FOLDER` to write a report to a different folder.

### Predicting from your own files

You can give any combination of the three sensors:

```bash
python scripts/predict.py --rgb examples/crown_1_rgb.png --hsi examples/crown_1_hsi.npy --lidar examples/crown_1_lidar.npy
python scripts/predict.py --rgb my_crown.png
```

| Sensor | Expected input |
|---|---|
| RGB | PNG, JPG or `.npy` image centred on one crown, about 16 m across (resized to 32 × 32) |
| Hyperspectral | `.npy` array of 369 bands × height × width (or a single 369-band spectrum), as reflectance |
| LiDAR | `.npy` canopy-height raster in metres (resized to 32 × 32) |

The `examples/` folder contains six ready-made test trees to try.

### Output

| Output | Contents |
|---|---|
| Terminal | Top-5 species with probabilities, predicted and true height and crown diameter, trust per sensor |
| `predictions/report.html` | Summary figures, trees-per-species table, table of all predictions and one card per tree |
| `predictions/cards/*.png` | One prediction card per tree |
| `predictions/predictions.csv` | All predictions in a spreadsheet-friendly format |

Each card shows the RGB image, the LiDAR height map, the hyperspectral spectrum, the top-5 species, the predicted height and crown diameter, and the weight given to each sensor.

---

## How the model works

```
 RGB image      ──►  RGB encoder (CNN)            ──►  128 numbers ─┐
 Hyperspectral  ──►  HSI encoder (spectral CNN)   ──►  128 numbers ─┼──►  trust per sensor  ──►  weighted fusion
 LiDAR          ──►  LiDAR encoder (PointNet)     ──►  128 numbers ─┘                               │
                                                                                                   ├──►  species (top-5)
                                                                                                   └──►  height, crown diameter
```

1. **Encoders.** Each sensor is turned into a 128-number summary by its own network.
2. **Quality estimator.** For each sensor, a small evidential network measures how much evidence that sensor provides. Little evidence means low trust. A missing sensor gets a trust of zero.
3. **Adaptive fusion.** The three summaries are combined as a weighted average, with weights proportional to trust.
4. **Heads.** One head predicts the species. A second head predicts height and crown diameter.
5. **Training.** The loss combines four parts:
   - focal loss for species, with class-balanced sampling so rare species are not ignored,
   - a cross-modal contrastive loss that pulls the three sensor summaries of the same tree together,
   - an evidential loss that trains the trust scores,
   - a regression loss for height and crown diameter.

   Sensors are randomly hidden during training (modality dropout).

| Component | File |
|---|---|
| Encoders | `src/models/encoders.py` |
| Quality estimator (trust) | `src/models/quality.py` |
| Adaptive fusion | `src/models/fusion.py` |
| Species and structure heads | `src/models/heads.py` |
| Full model | `src/models/model.py` |
| Losses | `src/losses/` |
| Training loop, sampler, modality dropout | `src/training/` |

---

## Results

### Real-species model (20 species)

The model was trained on 1,398 trees, tuned on 301 and tested on 301 unseen trees. The split was stratified by species, so every species appears in each part.

| Metric | Result |
|---|---|
| Species accuracy | **74.4%** (random guessing: 5%) |
| Macro-F1 | 0.739 |
| True species in the top 5 | **99.7%** |
| Height error (mean) | **0.90 m** (R² 0.98) |
| Crown-diameter error (mean) | **0.25 m** (R² 0.99) |

Accuracy with sensors removed (all 301 test trees):

| Sensors used | Accuracy |
|---|---|
| All three | 74% |
| Without RGB | 71% |
| Without hyperspectral | 70% |
| Without LiDAR | 59% |

With **whole forest sites held out** (SCBI and SERC never seen in training), species accuracy is 59.4%. This shows how much harder it is to generalise to a new forest.

Per-species F1 ranges from 1.00 (tulip poplar, quaking aspen) to 0.28 (sweetgum). Most errors are between similar-looking species, such as red maple and sweetgum, or eastern and western hemlock.

### Research experiments (167 species, site-held-out)

These results come from the main experiment pipeline. Test sites are MLBS, STEI and WREF (423 trees). The final model was chosen on validation data only. Full tables are in [`results/RESULTS.md`](results/RESULTS.md), and `results/results_page.html` is a visual summary.

| Metric | Baseline | Final model |
|---|---|---|
| Test accuracy | 0.693 | **0.846** (3 seeds: 0.849 ± 0.009) |
| Test macro-F1 | 0.508 | **0.570** |
| Accuracy, species seen in training | 0.757 | **0.925** |
| Accuracy with one sensor missing | 0.604 | **0.769** |
| Hyperspectral-only accuracy | 0.279 | **0.593** |
| Height / crown-diameter error | 2.11 m / 0.44 m | **1.74 m / 0.31 m** |

![Variants and ablations](results/figures/comparison.png)

### What the experiments show

1. **Lighting differences between sites were the main problem.**
   - In the baseline, removing the hyperspectral sensor actually *raised* accuracy, because that branch had learned each site's lighting.
   - Random brightness and colour changes during training fixed this and raised test accuracy by 15 points.

2. **Modality dropout makes missing sensors survivable.**
   - Without it, accuracy with one sensor missing falls by 3.3 points.
   - LiDAR-only accuracy falls from 0.567 to 0.461.

3. **The trust scores explain the model, but did not improve accuracy on this data.**
   - They detect hazy hyperspectral images almost perfectly (AUROC 0.99) and cut that sensor's weight from 0.47 to 0.05.
   - They do not detect blurred RGB or sparse LiDAR.
   - Over three seeds, plain averaging reaches the same accuracy (0.848 ± 0.021 vs 0.849 ± 0.009).

4. **The contrastive loss had no measurable effect.**

5. **Unseen species cannot be recognised.** 17 test species never occur at the training sites, so results are also reported on species seen in training.

![Accuracy with missing sensors](results/figures/missing_sensors.png)

---

## Datasets

### Real-species dataset

Created by `scripts/make_real_species_data.py` and saved as `data/real_species_tree_crowns.h5`.

| Key | Shape | Type | Content |
|---|---|---|---|
| `rgb` | (N, 3, 32, 32) | uint8 | Colour image, 0.5 m pixels (16 m window) |
| `hsi` | (N, 369, 8, 8) | float16 | Reflectance from 380 to 2510 nm, 2 m pixels |
| `lidar` | (N, 1, 32, 32) | float32 | Canopy height model in metres |
| `species`, `site` | (N,) | int64 | Indices into the `species_names` and `site_names` attributes |
| `height`, `crown_diameter` | (N,) | float32 | Metres |
| `degraded` | (N, 3) | bool | Whether RGB, HSI or LiDAR is degraded for this tree |
| `wavelengths` | (369,) | float64 | Band centres in nm |

**Species (20):**

- **Broadleaves:** red maple, sugar maple, northern red oak, white oak, turkey oak, American beech, yellow birch, tulip poplar, sweetgum, black gum, quaking aspen.
- **Conifers:** eastern white pine, longleaf pine, loblolly pine, lodgepole pine, eastern hemlock, western hemlock, balsam fir, Engelmann spruce, Douglas-fir.

**NEON sites (10):** HARV, BART, SCBI, SERC, MLBS, OSBS, TALL, UNDE, NIWO, WREF.

Each species only occurs at sites within its approximate natural range (see `src/data/species_catalog.py`).

![Real-species samples](results/figures/real_species_samples.png)

Options:

```bash
python scripts/make_real_species_data.py --num-samples 1000
python scripts/make_real_species_data.py --sites HARV BART SCBI SERC UNDE
```

### Research dataset

Created by `scripts/make_synthetic_data.py` and saved as `data/synthetic_neon_mmtc.h5`. It has 2,000 crowns, 167 species and 16 sites.

| Sensor | Stored as | Invalid values | Normalisation |
|---|---|---|---|
| RGB | (N, 3, 64, 64) uint8 | All-zero image means missing | Per-channel, fitted on training data |
| Hyperspectral | (N, 369, 12, 12) int16 × 10000 | `-9999`, NaN or saturated pixels are replaced; under 25% valid means missing | Per-band, fitted on training data |
| LiDAR | (N, 1024, 4) float32 point cloud | Invalid points are dropped; under 16 points means missing | Centred and scaled; 512 points per tree |

---

## Training and evaluation

### Train

```bash
# Real-species model (about 8 minutes on a CPU)
python scripts/train.py --config configs/real_species.yaml

# Research model
python scripts/train.py --config configs/default.yaml

# Change any setting from the command line
python scripts/train.py --config configs/real_species.yaml --set train.epochs=20

# Resume a stopped run
python scripts/train.py --resume outputs/last.pt
```

Checkpoints (`best.pt`, `last.pt`), logs and metrics are saved in the output folder set in the config.

### Evaluate

```bash
python scripts/evaluate.py --checkpoint results/real_species_model.pt --out predictions/evaluation
python scripts/evaluate.py --checkpoint results/final_model_best.pt
```

The evaluation reports:

- accuracy, top-5 accuracy, precision, recall, macro-F1 and weighted-F1,
- per-species F1 (`per_species.csv`) and a confusion matrix,
- expected calibration error (ECE),
- results with every combination of sensors removed,
- results with sensors silently corrupted,
- height and crown-diameter errors,
- plots in the `plots/` folder.

### Data splits

| Split mode | How it works | Used for |
|---|---|---|
| `stratified` | Each species is split separately: 70% train, 15% validation, 15% test | Real-species model |
| `site` | Whole forest sites go to train, validation or test | Research model, and the site-held-out comparison |

Set the mode with `split.mode` in the config.

---

## Reproducing all experiments

One command runs the full research study: three model variants, selection on validation data, three ablations, extra seeds and the final report.

```bash
python scripts/run_experiments.py           # about 3 hours on a CPU, much faster on a GPU
python scripts/run_experiments.py --quick   # 5-minute check that the pipeline runs
```

Results are written to `outputs/experiments/`. Seeds are fixed, so re-running reproduces the same numbers.

---

## Project structure

```
configs/
  default.yaml          settings for the research model
  real_species.yaml     settings for the real-species model
  experiments.yaml      variants and ablations for run_experiments.py

src/
  data/                 data generators, loaders, splits and cleaning
  models/               encoders, quality estimator, fusion, heads, full model
  losses/               focal, evidential, contrastive and combined losses
  training/             training loop, class-balanced sampler, modality dropout
  evaluation/           metrics, evaluation, plots, predictor and prediction report
  utils/                config loading, seeding, device selection, logging

scripts/
  make_real_species_data.py   create the 20-species dataset
  make_synthetic_data.py      create the 167-species dataset
  inspect_data.py             show the contents of a data file
  train.py                    train a model
  evaluate.py                 evaluate a trained model
  predict.py                  predict trees and build the HTML report
  run_experiments.py          run every experiment

tests/                  111 automated tests
results/                trained models, result tables and figures
examples/               six sample trees for predict.py
```

---

## Tests

```bash
python -m pytest -v
```

| Test file | What it checks |
|---|---|
| `test_dataset.py` | Data loading, handling of invalid values, normalisation fitted on training data only |
| `test_splits.py` | No forest site appears in more than one split |
| `test_encoders.py` | Output shapes, missing inputs, LiDAR point-order invariance |
| `test_fusion.py` | Missing sensors always get a weight of zero, for all sensor combinations |
| `test_losses.py`, `test_training.py` | Loss functions, balanced sampling, modality dropout |
| `test_smoke.py` | 2 training epochs on 64 trees finish in under 2 minutes; results are reproducible |
| `test_evaluation.py` | Metrics against hand-computed values and the full evaluation |
| `test_real_species.py` | The real-species dataset layout and species ranges |
| `test_predict.py` | The loader, the predictor and `predict.py` end to end |

---

## Limitations and future work

- **Simulated sensor data.** The next step is to train and evaluate on real NEON airborne data.
- **Real photos.** The model was trained on simulated crowns, so its predictions on real aerial photos are not meaningful yet.
- **Unknown species.** The model can only choose among the species it was trained on. Detecting unknown species (open-set recognition) is a possible extension.
- **Trust scores.** They make the model more explainable but did not improve accuracy here. Testing them on real sensor failures is future work.
- **Seeds.** Only the main comparison was run with three seeds; other results come from a single run.

---

## Requirements

Python 3.12, PyTorch 2.15.0, NumPy, h5py, PyYAML, matplotlib, Pillow and pytest. Exact versions are listed in `requirements.txt`.
