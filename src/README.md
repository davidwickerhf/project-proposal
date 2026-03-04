# Source Code Guide

This `src/` tree is the implementation scaffold for the full project pipeline.

## Module Map

- `common/`: immutable contracts (IDs, enums, file naming, canonical paths).
- `data/`: image standardization and manifest IO helpers.
- `embedding/`: encryption and embedding algorithms (currently deferred placeholders).
- `detection/`: detector implementations and SRM training/inference hooks (currently deferred placeholders).
- `evaluation/`: split generation and evaluation helpers.
- `pipeline/`: orchestration logic and CLI commands.

## Execution Flow

1. Initialize expected folders (`init-layout`).
2. Standardize raw images and produce `covers_master.csv` (`standardize-covers`).
3. Build payload artifacts manifest (`build-payload-manifest`).
4. Build stego job manifest (`build-stego-manifest`).
5. Create grouped 5-fold split (`create-splits`).
6. Build SRM training job manifest (`build-training-jobs`).
7. Run embedding stage (`run-embedding-stage` dry-run or execute).

## Design Rules

- `group_id` is the canonical unit across all modules.
- File naming/path contracts from `common/contracts.py` must not be bypassed.
- Keep image format fixed (`512x512`, RGB, PNG) across all branches.
- All stage outputs must be tracked via manifests.
- Deferred functions should raise `NotImplementedError` until implemented.
