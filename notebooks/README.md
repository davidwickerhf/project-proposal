# Notebooks Guide

This directory provides a notebook-first way to run the project pipeline, using the same functions and contracts as the CLI workflow.

## Prerequisites

- Run from project root (`/project-proposal`).
- Install dependencies:

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install jupyterlab ipykernel
```

- Start Jupyter:

```bash
jupyter lab
```

## Notebook Overview

1. `01_ml_generation_and_convergence.ipynb`
- Use case: ML cover generation and quick generation-quality/convergence checks.
- Main actions:
  - runs `generate_ml_covers_from_prompts(...)`
  - supports `engine='stub'` (fast) and `engine='diffusers'` (real models)
  - runs an inference-step sweep and plots a convergence proxy
- Typical outputs:
  - `data/manifests/covers_master_ml_a.csv`
  - `data/manifests/covers_master_ml_b.csv`
  - `data/manifests/covers_master_ml.csv`
  - `data/manifests/ml_generation_summary.json`

2. `02_embedding_and_cover_generation.ipynb`
- Use case: build and validate artifacts up to embedding.
- Main actions:
  - optional real download + optional ML generation + merge
  - runs payload/stego manifest creation
  - runs embedding stage (dry-run by default; executable mode optional)
- Typical outputs:
  - `data/manifests/covers_master.csv`
  - `data/manifests/payload_manifest.csv`
  - `data/manifests/stego_manifest.csv`

3. `03_training_workflow.ipynb`
- Use case: grouped-fold training setup and SRM training loop template.
- Main actions:
  - creates grouped 5-fold splits
  - builds per-method SRM training jobs (10 total)
  - defines SRM feature extraction + fold/method training-call pattern for deferred SRM functions
- Typical outputs:
  - `results/splits/splits_grouped5fold.json`
  - `results/splits/srm_training_jobs.csv`

4. `04_full_pipeline_end_to_end.ipynb`
- Use case: one notebook run for the full pipeline.
- Main actions:
  - optional data preparation/merge
  - runs `PipelineRunner.run_full_pipeline(...)`
  - computes metrics and optionally generates figures
  - displays output paths and key cardinality checks
  - supports SRM auto-train path during detector execution
- Typical outputs:
  - `results/predictions/predictions.csv`
  - `results/metrics/*.csv`
  - `results/figures/*.png`

## Recommended Run Order

1. `01_ml_generation_and_convergence.ipynb`
2. `02_embedding_and_cover_generation.ipynb`
3. `03_training_workflow.ipynb`
4. `04_full_pipeline_end_to_end.ipynb`

## Execution Modes

- Safe/default mode (works before deferred algorithms are implemented):
  - ML generation with `engine='stub'`
  - embedding and detectors in dry-run mode
  - SRM training in simulation/planning mode
- Full mode (after deferred functions are implemented):
  - ML generation with `engine='diffusers'`
  - enable embedding execution
  - enable detector execution
  - enable SRM feature extraction/training/inference calls

SRM auto-training behavior:
- In executable detector mode (`execute=True`, SRM enabled), the runner auto-trains SRM models per `(fold, method)` if no external SRM score provider is supplied.

## Notes for Collaborators

- Notebook cells call the same Python modules used by CLI scripts; notebooks are an alternative interface, not a separate pipeline.
- Artifacts are written to the same canonical `data/` and `results/` paths.
- Keep run flags explicit in notebook parameter cells to avoid accidental heavy runs.
