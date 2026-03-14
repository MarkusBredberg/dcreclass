# dcreclass

ML classification of diffuse radio emission in PSZ2 galaxy clusters. Processes multi-scale LOFAR radio images and trains CNN-based classifiers to distinguish diffuse emission (DE) from non-diffuse emission (NDE) sources.

## Overview

The pipeline proceeds in five steps:

1. **Download** — fetch PSZ2 FITS images from LOFAR surveys
2. **Categorise** — organise sources by classification label
3. **Process** — generate multi-scale cropped images (RAW + tapered + blurred)
4. **Train** — train a classifier on the processed images
5. **Evaluate** — plot metrics, ROC curves, and attention maps

## Installation

```bash
git clone <repo>
cd p2_DCRECLASS
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Requires Python ≥ 3.10.

## Data

PSZ2 FITS images are expected at `/users/mbredber/scratch/data/PSZ2/fits/` with a redshift table at `/users/mbredber/scratch/data/PSZ2/cluster_source_data.csv`.

Data classes:
- **50** — Diffuse Emission (DE)
- **51** — No Diffuse Emission (NDE)

## Scripts

### 01. Download

```bash
python scripts/01.rsync_PSZ2.py
```

Fetches FITS files from LOFAR surveys, checks completeness (RAW + T25/50/100kpc + SUB variants), and downloads in parallel.

### 02. Categorise

```bash
python scripts/02.categorise_PSZ2.py
```

Organises downloaded FITS by classification label.

### 03. Create Processed Images

Generates multi-scale cropped FITS and per-source montage PNGs. Each source produces up to 7 image versions: RAW, T25kpc, T50kpc, T100kpc, Blur25kpc, Blur50kpc, Blur100kpc.

```bash
python scripts/03.create_processed_images.py [OPTIONS]
```

| Argument | Default | Description |
|---|---|---|
| `--root` | `PSZ2/fits/` | Root directory of raw FITS files |
| `--z-csv` | `PSZ2/cluster_source_data.csv` | Redshift table |
| `--out` | `PSZ2/<mode>/montages/` | Montage output directory |
| `--fits-out` | `PSZ2/<mode>/fits_files/` | Processed FITS output directory |
| `--down` | `128,128` | Downsample size (H,W or C,H,W) |
| `--scales` | `25, 50, 100` | Taper/blur scales in kpc |
| `--crop-mode` | `beam_crop` | `beam_crop`, `fov_crop`, or `pixel_crop` |
| `--blur-method` | `circular` | `circular`, `circular_no_sub`, or `cheat` |
| `--fov-arcsec` | `300` | FOV in arcseconds (used with `fov_crop`) |
| `--n-workers` | auto | Number of parallel workers |
| `--only-one` | — | Debug: process a single named source |
| `--no-montage` | — | Skip montage PNG generation |
| `--comparison-plot` | — | Also produce a multi-source comparison grid |
| `--force` | — | Overwrite existing outputs |

**Crop modes:**
- `beam_crop` — crop in multiples of beam FWHM, equalised across all versions
- `fov_crop` — fixed arcsecond FOV (set with `--fov-arcsec`)
- `pixel_crop` — on-the-fly centre crop from raw files, no pre-processed FITS needed

**Blur methods:**
- `circular` — convolve RAW with circular Gaussian to match physical scale, subtract beam
- `circular_no_sub` — same but skip beam subtraction
- `cheat` — use FITS header beam directly

### 04. Train Classifier

```bash
python scripts/04.train_classifier.py [OPTIONS]
```

| Argument | Default | Description |
|---|---|---|
| `--classifier` | `ImageCNN` | `CNN`, `ScatterNet`, `DualCSN`, `DualSSN`, `ImageCNN` |
| `--versions` | `RAW` | `+`-separated image versions, e.g. `RAW` or `T25kpc+T50kpc` |
| `--crop-mode` | `beam_crop` | `beam_crop`, `beam_crop_no_sub`, `fov_crop`, `cheat_crop`, `pixel_crop` |
| `--blur-method` | `circular` | `circular`, `circular_no_sub`, `cheat` |
| `--folds` | `0` | Space-separated fold indices |
| `--num-experiments` | `2` | Number of random restarts per fold |
| `--lr` | `5e-5` | Learning rate |
| `--reg` | `1e-1` | Weight decay |
| `--label-smoothing` | `0.1` | Label smoothing factor |
| `--percentile-lo` / `--percentile-hi` | `30` / `99` | Normalisation percentile range |
| `--num-epochs-cuda` / `--num-epochs-cpu` | `200` / `100` | Max training epochs |
| `--patience` | `50` | Early stopping patience |
| `--run-dir` | scratch | Override output root for figures/logs |
| `--data-run-dir` | — | Override output root for models/metrics |
| `--force` | — | Overwrite existing caches |
| `--no-stretch` | — | Disable percentile stretch |
| `--no-augment` | — | Disable data augmentation |
| `--no-mixup` | — | Disable mixup |
| `--no-class-weights` | — | Disable class weighting |
| `--no-early-stopping` | — | Disable early stopping |
| `--no-scheduler` | — | Disable LR scheduler |

### 05. Plot Results

```bash
python scripts/05.plot_classifier_results.py [OPTIONS]
```

Same `--classifier`, `--version`, `--crop-mode`, `--blur-method`, `--folds`, `--lr`, `--reg`, `--run-dir`, `--data-run-dir` arguments as script 04. Also accepts `--no-attention` to skip attention map generation.

## Running on the HPC Cluster

Edit `jobs/classify.sh` to set `CLASSIFIER`, `CROP_MODE`, `VERSIONS`, `FOLDS`, and `RUN_DIR`, then submit:

```bash
sbatch jobs/classify.sh
```

The job runs both training (script 04) and evaluation (script 05) sequentially. Logs go to `outputs/logs/`.

## Package Structure

```
src/dcreclass/
├── data/
│   ├── loaders.py      # FITS loading, caching, augmentation, train/test split
│   └── processing.py   # Cropping, convolution kernels, NaN analysis
├── models/
│   └── classifiers.py  # CNN, ImageCNN, ScatterNet, DualCSN, DualSSN
├── training/
│   └── trainer.py      # Early stopping, mixup, metrics
└── utils/
    ├── fits.py          # FITS/WCS operations
    ├── calc_tools.py    # Normalisation, scattering coefficients
    ├── annotation.py    # Beam patches, scale bars
    └── plotting.py      # Visualisation
```

## Classifiers

| Name | Description |
|---|---|
| `ImageCNN` | Single-branch CNN with adaptive pooling |
| `CNN` | 6-layer conv net with batch norm and LeakyReLU |
| `ScatterNet` | Processes scattering transform coefficients |
| `DualCSN` | Dual-branch CNN with feature fusion |
| `DualSSN` | Dual-branch hybrid: CNN + scattering paths |
