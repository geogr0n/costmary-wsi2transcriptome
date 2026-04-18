# Reproducing the Main-Cohort Models

This note documents the main-cohort reproduction boundary and usage flow for
this repository. Its scope is deliberately narrow:

- it explains how to use the released checkpoints for the eight main models
- it points to the original upstream resources and access conditions
- it documents the **test-only** evaluation entry point in this repository

The public code repository for this package is:

- <https://github.com/geogr0n/costmary-wsi2transcriptome>

It does **not** re-explain upstream data preparation in detail. That part is
not reimplemented in this repository, but a minimal dummy preparation example is
provided under [examples/prepared_input_contract/](examples/prepared_input_contract).

## What This Repository Reproduces

`costmary` reproduces the **main-cohort training and evaluation stage** of the
paper once the inputs have already been prepared in the same preprocessing
contract used in the study.

More concretely, this repository is intended to cover:

- training and evaluation of the shared backbone/head configurations
- fold-wise prediction export on the main cohort
- pure test-only evaluation of released checkpoints on the main cohort and on
  any external cohort that follows the same prepared-input contract
- the main-cohort artifacts used to build the paper's controlled comparisons

This repository does **not** redistribute:

- raw GDC data
- upstream `SEQUOIA` preprocessing code or assets
- `UNI` weights or precomputed `UNI` features

## External Resources and Access Conditions

The released models in this repository sit on top of external upstream layers.
Those layers remain governed by their original sources, licenses, and access
conditions.

### `SEQUOIA`

- Repository: <https://github.com/gevaertlab/sequoia-pub>
- Paper: <https://www.nature.com/articles/s41467-024-54182-5>

`costmary` starts from **SEQUOIA-compatible prepared inputs**. The upstream
preprocessing logic itself is not redistributed here.

### `UNI`

- Model page: <https://huggingface.co/MahmoodLab/UNI>

The fixed feature space used in the paper comes from `UNI`. This repository does
not redistribute `UNI` weights or features. Users must obtain and use them
under the original provider terms.

### GDC

- GDC Portal: <https://portal.gdc.cancer.gov/>
- Access overview: <https://gdc.cancer.gov/access-data/data-access-processes-and-tools>
- Controlled-access documentation:
  <https://docs.gdc.cancer.gov/Encyclopedia/pages/Controlled_Access/>
- Controlled-access request guide:
  <https://gdc.cancer.gov/access-data/obtaining-access-controlled-data>

For GDC:

- **open-access** files can be downloaded directly under GDC rules
- **controlled-access** files require the corresponding `dbGaP` authorization
  and `eRA Commons` authentication

This repository does not bypass or replace those access conditions.

## Minimal Preparation Example and Data Handoff

To make the preparation boundary auditable, this repository now includes a
small **dummy prepared-data package**:

- [examples/prepared_input_contract/](examples/prepared_input_contract)

This package is intentionally modest. It contains:

- a schema-level example reference CSV
- a few synthetic HDF5 feature files that match the final `costmary` contract

It does **not** contain:

- real patient data
- real `UNI` features
- any real clustered feature HDF5 files
- any data that can be used to reproduce the paper's quantitative results on
  its own

Its role is only to show **how the handoff point is structured**.

### Concrete preparation workflow

For the real study, the main-cohort reproduction boundary is:

1. **Obtain lawful upstream data**
   - collect pathology and transcriptomic source data from GDC under the
     corresponding access conditions
   - open-access files can be downloaded directly; controlled-access files, if
     used, require the matching `dbGaP` authorization and `eRA Commons`
     authentication under GDC rules
2. **Apply SEQUOIA-compatible preprocessing**
   - use the public `SEQUOIA` repository and workflow to construct the matched
     reference tables and slide-level preparation contract used in the paper
3. **Extract slide features with UNI**
   - use the original `UNI` release under its original provider terms
4. **Cluster slide features into the final `costmary` contract**
   - produce one HDF5 file per slide
   - ensure that each HDF5 file exposes the dataset key `cluster_features`
5. **Run `costmary`**
   - training and evaluation start only after the previous steps are complete

### What the shipped dummy example shows

The local dummy package illustrates the **final input handoff** expected by
this repository:

- [examples/prepared_input_contract/ref_file_schema_example.csv](examples/prepared_input_contract/ref_file_schema_example.csv)
- [examples/prepared_input_contract/feature_root/](examples/prepared_input_contract/feature_root)

These files are synthetic and meaningless by design. They exist only to make
the expected naming, feature directory structure, and metadata contract
concrete.

### How to read the dummy example

The example reference CSV shows the minimum metadata fields that later become
the prepared table used by `costmary`:

- `patient_id`
- `tcga_project`
- `wsi_file_name`
- transcript columns with the `rna_` prefix

The synthetic HDF5 files show the actual directory and file structure consumed
by `costmary`, but they carry no biological meaning and are not intended for
training or quantitative evaluation.

### The final input contract expected by `costmary`

Once preprocessing and feature extraction are complete, the repository expects:

```text
<feature_path>/
  <tcga_project>/
    <wsi_file_name_without_.svs>/
      <wsi_file_name_without_.svs>.h5
```

Each HDF5 file must contain:

- `cluster_features`

This is the point at which the main-cohort training and evaluation code in this
repository takes over.

## Released Checkpoints

The eight main model configurations discussed in the paper are publicly
released in the following ModelScope repository:

- <https://www.modelscope.cn/models/geogron/costmary-wsi2transcriptome>

The table below records the correspondence between the paper names, the
`costmary` experiment names, and the public release location.

| Paper name | `costmary` experiment name | Checkpoint link |
| --- | --- | --- |
| `mean` | `mean` | <https://www.modelscope.cn/models/geogron/costmary-wsi2transcriptome> |
| `meanSPEX` | `mean_spex` | <https://www.modelscope.cn/models/geogron/costmary-wsi2transcriptome> |
| `HE2RNA` | `he2rna` | <https://www.modelscope.cn/models/geogron/costmary-wsi2transcriptome> |
| `HE2RNASPEX` | `he2rna_spex` | <https://www.modelscope.cn/models/geogron/costmary-wsi2transcriptome> |
| `SEQUOIA` route (`ViS` in code) | `vis` | <https://www.modelscope.cn/models/geogron/costmary-wsi2transcriptome> |
| `SEQUOIASPEX` / `ViSSPEX` | `vis_spex` | <https://www.modelscope.cn/models/geogron/costmary-wsi2transcriptome> |
| `SPD` | `spd` | <https://www.modelscope.cn/models/geogron/costmary-wsi2transcriptome> |
| `SPDSPEX` | `spd_spex` | <https://www.modelscope.cn/models/geogron/costmary-wsi2transcriptome> |

Two additional internal controls exist in the repository but are not part of
the eight released main-model checkpoints:

- `vis_d2`
- `projmean`

## Test-Only Evaluation Interface

To support transparent checkpoint reuse, this repository now includes a
**pure test-only entry point**:

- [evaluate_full_dataset.py](evaluate_full_dataset.py)

This script:

- loads a released checkpoint
- runs inference on a fully prepared cohort without retraining
- evaluates the checkpoint on the **full provided dataset**
- exports the same core statistics used by the paper's evaluation logic

The repository deliberately accepts **test-only evaluation on either the main
cohort or external cohorts**, provided that the supplied inputs follow the same
prepared reference-table and clustered-feature contract.

This is the intended interface for users who already have:

- a prepared reference table
- prepared clustered feature files
- a downloaded checkpoint

## How to Use the Released Models

### Step 1: download a checkpoint

Download the checkpoint corresponding to the model you want to test from its
public ModelScope release:

- <https://www.modelscope.cn/models/geogron/costmary-wsi2transcriptome>

The test-only script expects a local checkpoint path such as:

```text
/path/to/model_best.pt
```

### Step 2: run full-dataset evaluation

Example: evaluate the released `SEQUOIA` route with a linear head on a prepared
cohort.

```bash
python evaluate_full_dataset.py \
  --ref_file /path/to/ref_file.csv \
  --feature_path /path/to/features \
  --save_dir /path/to/output \
  --cohort TCGA \
  --exp_name vis \
  --checkpoint_path /path/to/model_best.pt
```

Example: evaluate the released `SPDSPEX` checkpoint.

```bash
python evaluate_full_dataset.py \
  --ref_file /path/to/ref_file.csv \
  --feature_path /path/to/features \
  --save_dir /path/to/output \
  --cohort HTMCP-LC \
  --exp_name spd_spex \
  --checkpoint_path /path/to/model_best.pt
```

For the eight standard released checkpoints, the script can infer the backbone
and decoder settings directly from `--exp_name`.

If you evaluate a non-standard checkpoint name, pass the architecture settings
explicitly, for example:

```bash
python evaluate_full_dataset.py \
  --ref_file /path/to/ref_file.csv \
  --feature_path /path/to/features \
  --save_dir /path/to/output \
  --cohort TCGA \
  --exp_name custom_checkpoint \
  --checkpoint_path /path/to/model_best.pt \
  --model_type vis \
  --head_type spex \
  --spex_k 16 \
  --depth 6 \
  --num_heads 16
```

## What the Test-Only Script Produces

For each evaluated checkpoint, the script writes to:

```text
<save_dir>/<cohort>/<exp_name>/
```

The key outputs are:

- `test_results.pkl`
  - full-dataset predictions, labels, random baseline predictions, slide names,
    and project labels
- `metadata.json`
  - evaluation metadata, checkpoint path, model type, head type, and parameter
    count
- `all_genes.csv`
  - per-gene full-dataset statistics
- `sig_genes.csv`
  - significant genes under the same filtering logic used by the paper's
    evaluation layer
- `summary_row.csv`
  - a one-row summary including all-gene PCC, top-1000 PCC, significant-gene
    count, and parameter count

The pure test-only script uses the same **gene-level evaluation logic** as the
paper's analysis layer, but it applies that logic to the full provided dataset
in one pass instead of aggregating over fold-held-out predictions.

## Practical Scope

The intended reproducibility claim for this repository is therefore:

> Given lawfully prepared inputs under the same preprocessing contract and a
> released checkpoint, `costmary` can reproduce the model-side inference and
> full-dataset evaluation stage used to audit the main configurations in the
> paper, both on the main cohort and on any external cohort prepared under the
> same contract.

That is the stage this repository is designed to make transparent and auditable.

## Third-Party Code and Licensing Boundary

This repository contains adapted code paths derived from public upstream
projects. In particular:

- the `SEQUOIA`-style path is adapted from `gevaertlab/sequoia-pub`
- the `HE2RNA`-style path is adapted from `owkin/HE2RNA_code`

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the current provenance
summary and licensing notes.
