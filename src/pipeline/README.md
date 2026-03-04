# `src/pipeline` Guide

`pipeline/` orchestrates all stages and is the main operational entrypoint.

## Files

- `config.py`
  - central defaults (image size, seeds, payload bits, delta, key IDs)
- `runner.py`
  - stage methods:
    - `init_layout`
    - `standardize_covers_from_index`
    - `build_payload_manifest`
    - `build_stego_manifest`
    - `create_grouped_splits`
    - `build_srm_training_jobs`
    - `run_embedding_stage`
    - `run_detector_stage`
    - `compute_metrics_from_predictions`
    - `generate_metrics_figures`
    - `run_full_pipeline`
- `cli.py`
  - command-line wrappers around runner stages

## Stage Responsibilities

- `init_layout`: create canonical `data/` and `results/` directories.
- `standardize_covers_from_index`: normalize images and emit `covers_master.csv`.
- `build_payload_manifest`: define payload artifacts and metadata.
- `build_stego_manifest`: enumerate all stego jobs.
- `create_grouped_splits`: build grouped fold partitions.
- `build_srm_training_jobs`: define per-method SRM training jobs per fold.
- `run_embedding_stage`: execute or dry-run embedding manifest rows.
- `run_detector_stage`: generate prediction table rows (folded, condition-aware).
- `compute_metrics_from_predictions`: aggregate metrics by fold/condition/source.
- `generate_metrics_figures`: generate AUC figures from metric tables.
- `run_full_pipeline`: orchestrate full experiment flow in one command.

## Closed-Loop Boundaries

- Deferred algorithm functions (encryption, embedding, statistical detectors, SRM train/score) must be pure in-memory functions.
- Deferred algorithm functions (encryption, embedding, statistical detectors, SRM feature/train/score) must be pure in-memory functions.
- Those functions must not read/write files directly.
- `pipeline/runner.py` owns all artifact I/O and converts between manifest paths and in-memory inputs.

## CLI Commands

```bash
python3 -m src.pipeline.cli --project-root . init-layout
python3 -m src.pipeline.cli --project-root . standardize-covers --input-index <raw_cover_index.csv>
python3 -m src.pipeline.cli --project-root . build-payload-manifest --covers-manifest data/manifests/covers_master.csv
python3 -m src.pipeline.cli --project-root . build-stego-manifest --covers-manifest data/manifests/covers_master.csv --payload-manifest data/manifests/payload_manifest.csv
python3 -m src.pipeline.cli --project-root . create-splits --covers-manifest data/manifests/covers_master.csv
python3 -m src.pipeline.cli --project-root . build-training-jobs --splits-json results/splits/splits_grouped5fold.json
python3 -m src.pipeline.cli --project-root . run-embedding-stage --stego-manifest data/manifests/stego_manifest.csv
python3 -m src.pipeline.cli --project-root . run-detectors --stego-manifest data/manifests/stego_manifest.csv --splits-json results/splits/splits_grouped5fold.json
python3 -m src.pipeline.cli --project-root . compute-metrics --predictions results/predictions/predictions.csv
python3 -m src.pipeline.cli --project-root . plot-metrics
python3 -m src.pipeline.cli --project-root . run-all --covers-manifest data/manifests/covers_master.csv --generate-figures
```

`run-embedding-stage` is dry-run by default. Add `--execute` only after embedding functions are implemented.
`run-detectors` is also dry-run by default. Add `--execute` only after detector functions are implemented.
When `run-detectors --execute` runs with SRM enabled and no external SRM score callback, the runner auto-trains SRM models per `(fold, method)` from train/val partitions before test scoring.
`run-all` is dry-run for embedding/detectors by default; add `--execute-embeddings --execute-detectors` only when deferred methods are implemented.
