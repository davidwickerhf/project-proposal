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

## CLI Commands

```bash
python3 -m src.pipeline.cli --project-root . init-layout
python3 -m src.pipeline.cli --project-root . standardize-covers --input-index <raw_cover_index.csv>
python3 -m src.pipeline.cli --project-root . build-payload-manifest --covers-manifest data/manifests/covers_master.csv
python3 -m src.pipeline.cli --project-root . build-stego-manifest --covers-manifest data/manifests/covers_master.csv --payload-manifest data/manifests/payload_manifest.csv
python3 -m src.pipeline.cli --project-root . create-splits --covers-manifest data/manifests/covers_master.csv
python3 -m src.pipeline.cli --project-root . build-training-jobs --splits-json results/splits/splits_grouped5fold.json
python3 -m src.pipeline.cli --project-root . run-embedding-stage --stego-manifest data/manifests/stego_manifest.csv
```

`run-embedding-stage` is dry-run by default. Add `--execute` only after embedding functions are implemented.
