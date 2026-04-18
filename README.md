# costmary-wsi2transcriptome

Public code repository for the controlled whole-slide-image to transcriptome
comparison reported in the accompanying paper.

This repository contains the trimmed training and evaluation package used to
compare several WSI backbones under a shared fixed-feature-space pipeline:

- `mean`
- `HE2RNA`
- `ViS` / `ViS-depth2`
- `projmean`
- `SPD`

Each compatible backbone can be trained with either:

- a standard linear prediction head
- a `SPEX` head that constrains predictions to a PCA subspace of the training
  transcriptome

The code assumes that slide features have already been extracted and clustered.
It does **not** perform feature extraction from raw WSIs. In practice,
`costmary-wsi2transcriptome` starts from **SEQUOIA-compatible prepared inputs**
rather than from raw pathology or raw transcriptomics.

## What This Repository Is For

This repository is meant for **model training, evaluation, and result export**
in a fixed feature space. It is useful when you want to:

- train a single backbone/head configuration
- reproduce the backbone sweep used in the paper
- compare linear heads and `SPEX` heads under the same fold split
- export fold-level predictions and metadata for downstream analysis
- run test-only evaluation on the main cohort or on any external cohort that
  follows the same prepared-input contract

## Repository Layout

```text
costmary-wsi2transcriptome/
|- examples/
|  `- prepared_input_contract/
|- train/
|  |- backbone/
|  |  |- he2rna_backbone.py
|  |  |- mean_backbone.py
|  |  |- projmean_backbone.py
|  |  |- spd_backbone.py
|  |  |- tformer_lin.py
|  |  `- __init__.py
|  |- heads.py
|  |- main.py
|  |- read_data.py
|  |- training.py
|  |- utils.py
|  `- __init__.py
|- evaluate_full_dataset.py
|- run_train.sh
|- REPRODUCING_MAIN_COHORT.md
|- requirements-linux.txt
|- LICENSE
`- THIRD_PARTY_NOTICES.md
```

## Main Components

### `train/main.py`

Main entry point for training and fold-wise evaluation.

It handles:

- argument parsing
- patient-level `k`-fold splitting
- optional `SPEX` PCA construction on the training fold
- model creation
- training / early stopping
- test-fold prediction export
- metadata export

### `train/backbone/`

Backbone implementations used in the controlled comparison:

- `tformer_lin.py`: `ViS` / SEQUOIA-style transformer path
- `he2rna_backbone.py`: HE2RNA-style aggregation path
- `spd_backbone.py`: covariance / SPD-based backbone
- `mean_backbone.py`: mean pooling baseline
- `projmean_backbone.py`: control model for the SPD path

### `train/heads.py`

Prediction heads:

- `make_layernorm_linear_head(...)`: standard linear decoder
- `SPEXHead`: predicts transcriptomes through a low-rank PCA subspace

### `train/read_data.py`

Dataset loader for precomputed clustered slide features and transcriptome
labels.

### `train/training.py`

Training loop, validation-based checkpointing, early stopping, and fold-level
evaluation.

### `train/utils.py`

Utilities for:

- patient-level `k`-fold splitting
- filtering slides without valid feature files
- custom collation that drops bad samples

### `run_train.sh`

Bash launcher for the experiment grid used in the paper. It runs:

- backbone-only experiments
- compatible `SPEX` variants for multiple `K` values

### `evaluate_full_dataset.py`

Pure test-only evaluation entry point for downloaded checkpoints and fully
prepared cohorts.

## Data Requirements

The code expects two inputs:

1. A CSV file with slide-level metadata and transcriptome targets
2. A feature root containing one HDF5 file per slide

These inputs are expected to follow the same preprocessing contract used by the
paper:

- matched reference tables rather than raw RNA-seq files
- transcriptome targets stored in `rna_` columns
- precomputed clustered `UNI` slide features stored as `cluster_features`

If you need the upstream preprocessing steps themselves, obtain them from the
appropriate upstream sources before using this repository.

### CSV Requirements

The reference CSV must contain at least:

- `patient_id`
- `tcga_project`
- `wsi_file_name`
- transcriptome columns named with the `rna_` prefix

Example transcriptome columns:

- `rna_TP53`
- `rna_EGFR`
- `rna_BRCA1`

### Feature Directory Layout

The feature root is expected to look like:

```text
<feature_path>/
`- <tcga_project>/
   `- <wsi_file_name>/
      `- <wsi_file_name>.h5
```

Each HDF5 file must contain:

- `cluster_features`

For non-GTEX inputs, `.svs` is stripped from the slide name before opening the
feature file.

## Installation

This repository currently ships a Linux-oriented requirements file:

```bash
pip install -r requirements-linux.txt
```

Core dependencies include:

- `torch`
- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `h5py`
- `einops`
- `tqdm`

## Quick Start

### Train a Single Backbone

Example: train the standard `ViS` backbone with a linear head.

```bash
python -m train.main \
  --ref_file /path/to/ref_file.csv \
  --feature_path /path/to/features \
  --save_dir /path/to/results \
  --cohort TCGA \
  --exp_name vis \
  --model_type vis \
  --head_type linear \
  --depth 6 \
  --num-heads 16 \
  --batch_size 16 \
  --num_epochs 200 \
  --train
```

### Train a `SPEX` Variant

Example: train `ViS` with a `SPEX` head using `K=16`.

```bash
python -m train.main \
  --ref_file /path/to/ref_file.csv \
  --feature_path /path/to/features \
  --save_dir /path/to/results \
  --cohort TCGA \
  --exp_name vis_spex \
  --model_type vis \
  --head_type spex \
  --spex_k 16 \
  --depth 6 \
  --num-heads 16 \
  --batch_size 16 \
  --num_epochs 200 \
  --train
```

### Run the Full Backbone Sweep

```bash
bash run_train.sh \
  --ref_file /path/to/ref_file.csv \
  --feature_path /path/to/features \
  --save_dir /path/to/results \
  --cohort TCGA
```

The launcher runs:

- linear-head experiments for `vis`, `vis_d2`, `spd`, `he2rna`, `mean`,
  `projmean`
- `SPEX` experiments for all compatible backbones
- `SPEX` subspace sizes `K in {8, 16, 32, 64}`

`vis_d2` is implemented by calling the `vis` model with `--depth 2`.

### Pure Test-Only Evaluation With Released Checkpoints

The eight principal released checkpoints are publicly available at:

- <https://www.modelscope.cn/models/geogron/costmary-wsi2transcriptome>

The reproduction note is available at:

- [REPRODUCING_MAIN_COHORT.md](REPRODUCING_MAIN_COHORT.md)

To evaluate a downloaded checkpoint on a fully prepared cohort without
retraining, use:

```bash
python evaluate_full_dataset.py \
  --ref_file /path/to/ref_file.csv \
  --feature_path /path/to/features \
  --save_dir /path/to/output \
  --cohort TCGA \
  --exp_name vis \
  --checkpoint_path /path/to/model_best.pt
```

For the eight standard released checkpoints, `evaluate_full_dataset.py` can
infer the backbone/head settings directly from `--exp_name`.

This test-only path is intended for:

- the main TCGA cohort
- any external cohort that has been prepared into the same reference-table and
  clustered-feature contract

## Key Training Behavior

### Cross-validation

The code uses patient-level `k`-fold splitting via `patient_kfold(...)`.

Default behavior:

- `k = 5`
- patient-level folds
- within each training pool, a validation split is created with
  `valid_size = 0.1`

### `SPEX`

When `--head_type spex` is used:

- PCA is fit on the training-fold transcriptome matrix
- the head predicts coefficients in a `K`-dimensional subspace
- predictions are reconstructed as:

```text
y = mu + z @ U_k^T
```

### Mean backbone shortcut

For `model_type=mean`, mean pooled inputs are precomputed once per fold for
faster training.

### Random baseline at evaluation time

After evaluating the trained model on the test fold, the script also evaluates
a randomly initialized model of the same backbone family. Those predictions are
stored alongside the trained model outputs in `test_results.pkl`.

## Output Files

Each experiment is written to:

```text
<save_dir>/<cohort>/<exp_name>/
```

Typical contents:

- `train_0.npy`, `val_0.npy`, `test_0.npy`, ...: patient IDs per fold
- `model_best.pt`, `model_best_1.pt`, ...: best fold checkpoints
- `test_results.pkl`: fold-wise predictions, labels, random baseline outputs,
  slide names, and project names
- `metadata.json`: experiment metadata such as:
  - model type
  - head type
  - `spex_k`
  - number of folds
  - backbone parameter count
  - per-fold training time
  - average / total fold training time
- `all_genes.csv`, `sig_genes.csv`, `summary_row.csv`:
  - available when using `evaluate_full_dataset.py`
  - full-dataset evaluation exports for released checkpoints

## Important CLI Arguments

### Required

- `--ref_file`
- `--feature_path`
- `--save_dir`

### Backbone / head selection

- `--model_type {vis, he2rna, spd, mean, projmean}`
- `--head_type {linear, spex}`
- `--spex_k`

### ViS settings

- `--depth`
- `--num-heads`

### SPD / projmean settings

- `--spd_r`
- `--spd_eps`
- `--spd_latent_dim`
- `--spd_mlp_depth`
- `--spd_dropout`

### HE2RNA settings

- `--he2rna_ks`
- `--he2rna_dropout`

### Training settings

- `--seed`
- `--lr`
- `--batch_size`
- `--num_epochs`
- `--patience`
- `--k`
- `--save_on`
- `--stop_on`

## Reproducibility Notes

The training entry point sets:

- NumPy seed
- PyTorch seed
- Python `random` seed
- deterministic CuDNN settings

The dataloader worker seeds are also initialized explicitly.

## Limitations

- This package assumes that feature extraction and clustering have already been
  completed.
- `run_train.sh` is a Bash launcher; on Windows, you may prefer calling
  `python -m train.main` directly.
- The provided requirements file is Linux-oriented.

## License and Provenance

This repository is distributed under **GNU GPL v3**. See:

- [LICENSE](LICENSE)
- [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)

Some components were adapted from public upstream projects, including:

- `gevaertlab/sequoia-pub`
- `owkin/HE2RNA_code`

Please read `THIRD_PARTY_NOTICES.md` before redistributing modified versions.
