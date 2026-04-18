# Minimal Input-Contract Example

This directory provides a **dummy input-contract example** for the `costmary`
reproduction materials.

It is intentionally limited:

- the files here are **synthetic and non-biological**
- they are included only to show **how the final input contract is organized**
- they do **not** reproduce the paper's quantitative results
- they do **not** replace the original upstream preparation workflow

The real study depends on upstream steps that are **not** implemented in this
repository:

1. lawful GDC data acquisition
2. SEQUOIA-compatible preprocessing
3. `UNI` feature extraction
4. clustering of slide features into the HDF5 format consumed by `costmary`

This example package therefore starts **after** those upstream steps. It
contains:

- a schema-level reference CSV
- a small synthetic feature root with fake HDF5 files

It does **not** contain:

- raw GDC data
- protected or controlled-access data
- `UNI` weights
- real `UNI` features
- any real clustered feature HDF5 files used by the study

## Directory Layout

```text
examples/prepared_input_contract/
  ref_file_schema_example.csv
  feature_root/
    BRCA/
      TCGA-BRCA-FAKE-0001-01A-01Z-DX1/
        TCGA-BRCA-FAKE-0001-01A-01Z-DX1.h5
    COAD/
      TCGA-COAD-FAKE-0003-01A-01Z-DX1/
        TCGA-COAD-FAKE-0003-01A-01Z-DX1.h5
    LUAD/
      TCGA-LUAD-FAKE-0002-01A-01Z-DX1/
        TCGA-LUAD-FAKE-0002-01A-01Z-DX1.h5
```

## How To Interpret This Example

### `ref_file_schema_example.csv`

This file illustrates the **minimum metadata contract** that later becomes the
prepared reference table used by `costmary`:

- `patient_id`
- `tcga_project`
- `wsi_file_name`
- transcriptome columns with the `rna_` prefix

The values are synthetic and only illustrate naming conventions and column
structure.

### `feature_root/`

These folders illustrate the actual **handoff point consumed by `costmary`**.
Each synthetic slide directory contains one fake HDF5 file whose internal
dataset key matches the real contract:

- `cluster_features`

The tensors inside these HDF5 files are meaningless synthetic arrays. They are
included only so users can inspect the expected directory layout and file
structure without accessing real data.

## What `costmary` Actually Consumes

For training or test-only evaluation, the repository expects:

- a prepared CSV reference table
- a feature root containing one HDF5 file per slide
- dataset key `cluster_features` inside each HDF5 file

That final feature contract is documented in:

- [README.md](../../README.md)
- [REPRODUCING_MAIN_COHORT.md](../../REPRODUCING_MAIN_COHORT.md)

## Legal Boundary

This dummy package is safe to redistribute because it contains no real patient
data and no third-party model assets. Real reproductions remain subject to:

- GDC data access conditions
- the original `SEQUOIA` repository and preprocessing terms
- the original `UNI` access and usage terms
