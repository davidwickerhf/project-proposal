# Source Code Guide

This `src/` tree now reflects the final proposal only.

## Module Map

- `common/`: shared enums, filenames, and canonical paths
- `data/`: grayscale standardization, dual cover export, manifest helpers
- `embedding/`: AES, spatial LSB, and JPEG DCT-LSB stubs
- `detection/`: primary statistical-detector stubs and optional `SRNet` extension stub
- `evaluation/`: grouped splits, metrics, and figures
- `pipeline/`: orchestration and CLI

## Main Execution Flow

1. Standardize raw images into grayscale PNG and JPEG carrier variants.
2. Build payload manifest for `low/medium/high x plain/encrypted`.
3. Build stego manifest for `lsb` and `dct`.
4. Create grouped 5-fold splits by `group_id`.
5. Run embedding stage.
6. Run detector stage.
7. Aggregate metrics and figures.

## Design Rules

- `group_id` is the canonical matching unit.
- Covers are stored twice: PNG for spatial, JPEG Q=95 for frequency.
- Mainline detectors are training-free.
- SRNet is optional and separate from confirmatory analyses.
- Deferred functions stay pure in-memory; the runner owns file I/O.
