# Does the Source of Carrier Image Affect Steganographic Detectability?

Comparative study of steganographic detectability on real vs ML-generated image carriers.

**Course:** Project 2.2, Department of Advanced Computing Sciences, Maastricht University  
**Team:** Abdul Moiz Akbar, Malo Coquin, Daria Gjonbalaj, Nico Muller-Spath, Jimena Narvaez del Cid, David Wicker, Nikolas Zouros  
**Date:** February 2026

## Quick Links

- Midway proposal (final, PDF): [docs/proposals/midway_proposal_final.pdf](docs/proposals/midway_proposal_final.pdf)
- Midway proposal slides (full, PDF): [docs/slides/midway_proposal_slides.pdf](docs/slides/midway_proposal_slides.pdf)
- Implementation slides (PDF): [docs/slides/implementation_slides.pdf](docs/slides/implementation_slides.pdf)

## Table of Contents

- [Quick Links](#quick-links)
- [Repository Structure](#repository-structure)
- [Technical Implementation Plan](#technical-implementation-plan)
- [1) Project Scope and Locked Design](#1-project-scope-and-locked-design)
- [2) Repository Implementation Status](#2-repository-implementation-status)
- [3) Data Contracts for Collaborators](#3-data-contracts-for-collaborators)
- [4) Required Manifest Schemas](#4-required-manifest-schemas)
- [5) Input Needed to Populate Covers](#5-input-needed-to-populate-covers)
- [6) End-to-End Runbook](#6-end-to-end-runbook)
- [7) Deferred Function Specifications](#7-deferred-function-specifications)
- [8) Training and Fold Logic](#8-training-and-fold-logic)
- [9) Detector Execution Policy](#9-detector-execution-policy)
- [10) Analysis Workflow After Pipeline Completion](#10-analysis-workflow-after-pipeline-completion)
- [11) Recommended Final-Report AUC Figures](#11-recommended-final-report-auc-figures)
- [12) Multi-Run Workflow](#12-multi-run-workflow)
- [13) Validation and Integrity Checklist](#13-validation-and-integrity-checklist)
- [14) CLI Command Reference](#14-cli-command-reference)
- [To Do List](#to-do-list)

## Repository Structure

```text
project-proposal/
├── README.md
├── docs/
│   ├── proposals/
│   │   ├── midway_proposal_final.tex
│   │   └── midway_proposal_final.pdf
│   └── slides/
│       ├── midway_proposal_slides.tex
│       ├── midway_proposal_slides.pdf
│       ├── implementation_slides.tex
│       ├── implementation_slides.pdf
│       ├── midway_proposal_slides_5min.tex
│       └── midway_proposal_slides_5min.pdf
├── src/
│   ├── common/
│   ├── data/
│   ├── embedding/
│   ├── detection/
│   ├── evaluation/
│   └── pipeline/
├── data/
├── results/
└── notebooks/
```

## Technical Implementation Plan

This section is the implementation source-of-truth for collaborators. If README conflicts with any older file-level notes, README wins.

### 1) Project Scope and Locked Design

Core comparison objective:
- Measure detectability differences by source (`real`, `ml_a`, `ml_b`) under matched embedding/payload/encryption settings.

Locked data/model design:
- Real data: `COCO` + `Flickr30k` only.
- ML data: `SDXL` (`ml_a`) + `PixArt-alpha` (`ml_b`) only.
- Single image format across the full pipeline: `512x512`, RGB, 8-bit PNG.
- Canonical unit is `group_id` (one caption/prompt anchor).
- Total groups: `500`.
- Covers per group: `3` (`real`, `ml_a`, `ml_b`) => `1,500` covers total.
- Stego variants per cover: `2 methods x 3 payloads x 2 encryption = 12`.
- Total stegos: `1,500 x 12 = 18,000`.

### 2) Repository Implementation Status

Already implemented scaffolding:
- `src/common/contracts.py`: contracts for IDs, names, and canonical paths.
- `src/data/images.py`: image loading, center-crop, resize, PNG save.
- `src/data/manifests.py`: CSV/JSON manifest read/write helpers.
- `src/evaluation/splits.py`: grouped 5-fold generation by `group_id`.
- `src/pipeline/config.py`: pipeline configuration defaults.
- `src/pipeline/runner.py`: stage orchestration and artifact manifest generation.
- `src/pipeline/cli.py`: CLI entrypoint to run stages.

Deferred functions (intentionally not implemented yet):
- `src/embedding/encryption.py`
- `src/embedding/lsb.py`
- `src/embedding/dct.py`
- `src/detection/statistical.py`
- `src/detection/srm.py`

### 3) Data Contracts for Collaborators

Directory contract:

```text
data/
├── covers/{real|ml_a|ml_b}/
├── payloads/{plain|encrypted}/{low|medium|high}/
├── stego/{lsb|dct}/{low|medium|high}/{plain|encrypted}/{real|ml_a|ml_b}/
└── manifests/

results/
├── predictions/
├── metrics/
├── splits/
└── figures/
```

Filename contract:
- Cover: `g{group_id:04d}__src-{source}.png`
- Stego: `g{group_id:04d}__src-{source}__m-{method}__p-{payload}__e-{encryption}.png`
- Payload: `g{group_id:04d}__p-{payload}__e-{encryption}.bin`

Allowed values:
- `source in {real, ml_a, ml_b}`
- `method in {lsb, dct}`
- `payload in {low, medium, high}`
- `encryption in {plain, encrypted}`

### 4) Required Manifest Schemas

`data/manifests/covers_master.csv`
- `group_id`, `source`, `dataset`, `orig_id`, `caption_id`, `caption_text`, `image_path`, `qc_pass`, `qc_score`, `seed`

`data/manifests/payload_manifest.csv`
- `group_id`, `payload_level`, `encryption`, `payload_path`, `payload_bits`, `aes_iv`, `aes_key_id`, `seed`

`data/manifests/stego_manifest.csv`
- `group_id`, `source`, `method`, `payload_level`, `encryption`, `cover_path`, `payload_path`, `stego_path`, `embed_params`, `seed`

`results/splits/splits_grouped5fold.json`
- per fold: `train_group_ids`, `val_group_ids`, `test_group_ids`

Predictions tables (`results/predictions/*.csv`)
- `fold`, `detector`, `group_id`, `source`, `method`, `payload_level`, `encryption`, `label`, `score`

### 5) Input Needed to Populate Covers

To standardize and register covers, collaborators provide one CSV input index for:
- selected real images,
- generated `ml_a` images,
- generated `ml_b` images.

Required columns for `standardize-covers` input CSV:
- `group_id`
- `source`
- `dataset`
- `orig_id`
- `caption_id`
- `caption_text`
- `raw_image_path`
- `qc_pass`
- `qc_score`
- `seed`

The pipeline reads `raw_image_path`, standardizes to canonical PNG, and writes `covers_master.csv`.

### 6) End-to-End Runbook

Run from repo root:

```bash
# Install dependencies
python3 -m pip install -r requirements.txt

# 0) Build real covers and prompts (already done once in this repo)
python3 -m src.data.download_real_covers --project-root .

# 1) Generate ML covers from generation prompts
python3 -m src.data.generate_ml_covers --project-root . --prompts-csv data/manifests/generation_prompts.csv --engine diffusers

# 2) Merge real + ml_a + ml_b into final covers_master.csv
python3 -m src.data.merge_covers_master --project-root . --real-manifest data/manifests/covers_master_real.csv --ml-a-manifest data/manifests/covers_master_ml_a.csv --ml-b-manifest data/manifests/covers_master_ml_b.csv --output-manifest data/manifests/covers_master.csv --expected-groups 500

# 3) Pipeline stages
python3 -m src.pipeline.cli --project-root . init-layout
python3 -m src.pipeline.cli --project-root . build-payload-manifest --covers-manifest data/manifests/covers_master.csv
python3 -m src.pipeline.cli --project-root . build-stego-manifest --covers-manifest data/manifests/covers_master.csv --payload-manifest data/manifests/payload_manifest.csv
python3 -m src.pipeline.cli --project-root . create-splits --covers-manifest data/manifests/covers_master.csv
python3 -m src.pipeline.cli --project-root . build-training-jobs --splits-json results/splits/splits_grouped5fold.json
python3 -m src.pipeline.cli --project-root . run-embedding-stage --stego-manifest data/manifests/stego_manifest.csv

# 4) Detection + aggregation
python3 -m src.pipeline.cli --project-root . run-detectors --stego-manifest data/manifests/stego_manifest.csv --splits-json results/splits/splits_grouped5fold.json
python3 -m src.pipeline.cli --project-root . compute-metrics --predictions results/predictions/predictions.csv
python3 -m src.pipeline.cli --project-root . plot-metrics

# One-command orchestration (same stages as above)
python3 -m src.pipeline.cli --project-root . run-all --covers-manifest data/manifests/covers_master.csv --generate-figures
```

Execution notes:
- `run-embedding-stage` is dry-run by default (counts rows).
- `run-detectors` is dry-run by default (writes planned prediction rows with empty scores).
- Adding `--execute` invokes embedding placeholders and will fail until those functions are implemented.
- Adding `--execute` to `run-detectors` invokes detector placeholders and will fail until those functions are implemented.
- `build-payload-manifest --write-files` will invoke encryption placeholder for encrypted payloads.
- `run-all` is dry-run by default for embedding/detectors. Use `--execute-embeddings --execute-detectors` only after deferred functions are implemented.

### 7) Deferred Function Specifications

Contract rule for all deferred functions:
- Closed-loop only: receive in-memory inputs, return in-memory outputs.
- No file reads/writes inside these functions.
- Pipeline runner handles all disk I/O and manifest coordination.

`encrypt_payload_aes_256_cbc(payload: bytes, key: bytes, iv: bytes) -> bytes`
- Input: plaintext payload bytes, 32-byte key, 16-byte IV.
- Output: ciphertext bytes.
- Side effects: none.

`decrypt_payload_aes_256_cbc(ciphertext: bytes, key: bytes, iv: bytes) -> bytes`
- Input: ciphertext bytes, matching key/IV.
- Output: plaintext bytes.
- Side effects: none.

`embed_lsb(cover_image: PIL.Image.Image, payload_bytes: bytes, payload_level: str, prng_key: int) -> PIL.Image.Image`
- Input: canonical cover image and payload bytes.
- Output: stego image in memory.
- Side effects: none.

`embed_dct_qim(cover_image: PIL.Image.Image, payload_bytes: bytes, payload_level: str, delta: float) -> PIL.Image.Image`
- Input: canonical cover image and payload bytes.
- Output: stego image in memory.
- Side effects: none.

`rs_analysis_score(image: PIL.Image.Image) -> float`
`chi_square_score(image: PIL.Image.Image) -> float`
`block_dct_shift_score(image: PIL.Image.Image) -> float`
- Input: in-memory image.
- Output: scalar detector score.
- Side effects: none.
- Applicability: `RS`/`chi-square` for LSB branch, `block_dct_shift` for DCT branch.

`train_srm_ec_model(training_input: SRMTrainingInput) -> SRMModelArtifact`
- Input: one method/fold in-memory train/validation feature-label split.
- Output: in-memory trained model artifact.
- Side effects: none.

`score_srm_ec_model(model: SRMModelArtifact, x_samples: Sequence[Sequence[float]]) -> list[float]`
- Input: trained in-memory model + in-memory feature rows.
- Output: one score per sample.
- Side effects: none.

### 8) Training and Fold Logic

Why folds:
- Folds are required for trained models (SRM) and used consistently for all detectors to keep comparisons fair.

Split protocol:
- Grouped 5-fold by `group_id` (prevents leakage across `real/ml_a/ml_b` variants of the same semantic anchor).
- Per fold:
  - Test groups: `100`
  - Remaining `400` split into train `350` + validation `50`

Per-fold counts:
- Train covers: `350 x 3 = 1,050`
- Validation covers: `50 x 3 = 150`
- Test covers: `100 x 3 = 300`

Per method, positives per fold (pooled payload+encryption):
- Train: `1,050 x 6 = 6,300`
- Validation: `150 x 6 = 900`
- Test: `300 x 6 = 1,800`

SRM training scope:
- Train SRM per method, per fold.
- `2 methods x 5 folds = 10` SRM training runs total.
- This is generated as `results/splits/srm_training_jobs.csv`.

### 9) Detector Execution Policy

Confirmatory detectors:
- `SRM+EC` (trained)
- `RS` (LSB confirmatory)
- `Block-DCT shift test` (DCT confirmatory)

Sensitivity-only detector:
- `chi-square` (LSB only, not confirmatory)

Applicability:
- LSB rows: `SRM+EC`, `RS`, `chi-square`
- DCT rows: `SRM+EC`, `Block-DCT shift test`

### 10) Analysis Workflow After Pipeline Completion

1. Build out-of-fold prediction tables from test partitions only.
2. Compute fold-level AUC and secondary metrics by detector/condition.
3. Aggregate across folds (mean + intervals).
4. Compute fold-level contrasts to answer RQs.

RQ mapping:
- RQ1: `AUC(real) - AUC(pooled ML)`
- RQ2: `AUC(ml_a) - AUC(ml_b)`
- RQ3: source-effect change across payload levels
- RQ4: source-effect change across embedding methods
- RQ5: source-effect change across encryption states

### 11) Recommended Final-Report AUC Figures

Main paper set:
- Figure 1 (RQ1 + RQ2): detector-wise AUC comparisons with fold CI.
- Figure 2 (RQ3): payload trend plot of source contrasts.
- Figure 3 (RQ4 + RQ5): method interaction and encryption interaction plots.
- Appendix: full ROC curves per detector/condition/fold summary.

### 12) Multi-Run Workflow

Current scaffold writes canonical output paths. For repeated experiments:
- Use a run label and archive outputs after each run.
- Recommended convention:
  - `results/runs/<run_id>/...`
  - `data/manifests/runs/<run_id>/...`
- Do not overwrite previous finalized run artifacts.

### 13) Validation and Integrity Checklist

Before claiming pipeline completion:
- `500` groups exist in covers manifest.
- `1,500` covers exist (`3` per group).
- `18,000` stego records exist (`12` per cover).
- Every stego maps to one cover by `group_id + source`.
- No split leakage (`group_id` appears in only one split partition per fold).
- Detector applicability rules are enforced.
- SRM jobs equal `10` total (`2 methods x 5 folds`).
- Manifests include seeds and parameter snapshots for reproducibility.

### 14) CLI Command Reference

Data scripts:

```bash
# Download real covers + prompt manifest
python3 -m src.data.download_real_covers --project-root .

# Generate ML covers (SDXL for ml_a, PixArt-alpha for ml_b)
python3 -m src.data.generate_ml_covers --project-root . --prompts-csv data/manifests/generation_prompts.csv --engine diffusers

# Generate ML covers in deterministic stub mode (fast smoke runs)
python3 -m src.data.generate_ml_covers --project-root . --prompts-csv data/manifests/generation_prompts.csv --engine stub --max-groups 25

# Merge source manifests into final covers_master.csv
python3 -m src.data.merge_covers_master --project-root . --real-manifest data/manifests/covers_master_real.csv --ml-a-manifest data/manifests/covers_master_ml_a.csv --ml-b-manifest data/manifests/covers_master_ml_b.csv --output-manifest data/manifests/covers_master.csv --expected-groups 500
```

Pipeline CLI (`src.pipeline.cli`) subcommands:

```bash
# Setup
python3 -m src.pipeline.cli --project-root . init-layout
python3 -m src.pipeline.cli --project-root . standardize-covers --input-index data/manifests/raw_cover_index_real.csv

# Manifest planning/build
python3 -m src.pipeline.cli --project-root . build-payload-manifest --covers-manifest data/manifests/covers_master.csv
python3 -m src.pipeline.cli --project-root . build-payload-manifest --covers-manifest data/manifests/covers_master.csv --write-files
python3 -m src.pipeline.cli --project-root . build-stego-manifest --covers-manifest data/manifests/covers_master.csv --payload-manifest data/manifests/payload_manifest.csv
python3 -m src.pipeline.cli --project-root . create-splits --covers-manifest data/manifests/covers_master.csv
python3 -m src.pipeline.cli --project-root . build-training-jobs --splits-json results/splits/splits_grouped5fold.json

# Embedding stage
python3 -m src.pipeline.cli --project-root . run-embedding-stage --stego-manifest data/manifests/stego_manifest.csv
python3 -m src.pipeline.cli --project-root . run-embedding-stage --stego-manifest data/manifests/stego_manifest.csv --execute

# Detector stage
python3 -m src.pipeline.cli --project-root . run-detectors --stego-manifest data/manifests/stego_manifest.csv --splits-json results/splits/splits_grouped5fold.json
python3 -m src.pipeline.cli --project-root . run-detectors --stego-manifest data/manifests/stego_manifest.csv --splits-json results/splits/splits_grouped5fold.json --execute
python3 -m src.pipeline.cli --project-root . run-detectors --stego-manifest data/manifests/stego_manifest.csv --splits-json results/splits/splits_grouped5fold.json --disable-srm --skip-unimplemented

# Metrics stage
python3 -m src.pipeline.cli --project-root . compute-metrics --predictions results/predictions/predictions.csv
python3 -m src.pipeline.cli --project-root . compute-metrics --predictions results/predictions/predictions.csv --quality-metrics-input results/metrics/quality_input.csv

# Figure stage
python3 -m src.pipeline.cli --project-root . plot-metrics
python3 -m src.pipeline.cli --project-root . plot-metrics --metrics-dir results/metrics --figures-dir results/figures

# Full orchestration in one command
python3 -m src.pipeline.cli --project-root . run-all --covers-manifest data/manifests/covers_master.csv
python3 -m src.pipeline.cli --project-root . run-all --covers-manifest data/manifests/covers_master.csv --generate-figures
python3 -m src.pipeline.cli --project-root . run-all --covers-manifest data/manifests/covers_master.csv --execute-embeddings --execute-detectors --generate-figures
python3 -m src.pipeline.cli --project-root . run-all --covers-manifest data/manifests/covers_master.csv --execute-embeddings --execute-detectors --disable-srm --skip-unimplemented --generate-figures
```

## To Do List

### Priority Pending Work
- [ ] Implement `encrypt_payload_aes_256_cbc` and `decrypt_payload_aes_256_cbc` with validation and deterministic behavior.
- [ ] Implement `embed_lsb` according to payload-level policy and deterministic PRNG ordering.
- [ ] Implement `embed_dct_qim` with payload-level coefficient policy and deterministic behavior.
- [ ] Implement statistical detectors: `rs_analysis_score`, `chi_square_score`, `block_dct_shift_score`.
- [ ] Implement SRM functions: `train_srm_ec_model` and `score_srm_ec_model`.
- [ ] Run ML cover generation with real models (`ml_a` via SDXL and `ml_b` via PixArt-alpha) from `generation_prompts.csv`.
- [ ] Run real+ML merge to produce final `data/manifests/covers_master.csv` (`1,500` covers total).
- [ ] Materialize payload binaries and stego images end-to-end for full matrix (`18,000` stegos).
- [ ] Execute detector stage to produce populated prediction tables under `results/predictions/`.
- [ ] Execute metrics stage and validate fold/condition/source outputs under `results/metrics/`.
- [ ] Generate final analysis figures/tables that explicitly answer RQ1–RQ5.

### Current Foundation Status (Brief)
- [x] Pipeline structure, manifest contracts, and grouped split scaffolding are in place.
- [x] Real-image set and prompt manifest are prepared.
- [x] Test scaffolding exists to guide the remaining implementations.
