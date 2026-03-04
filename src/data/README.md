# `src/data` Guide

`data/` contains utilities for image handling and manifest persistence.

## Files

- `images.py`
  - load image
  - center-crop to square
  - standardize to `512x512` RGB
  - save PNG
- `manifests.py`
  - dataclass records for covers/payloads/stegos/training jobs
  - CSV/JSON read/write helpers
- `download_real_covers.py`
  - build real-image cover set + prompt manifest from COCO/Flickr30k
- `generate_ml_covers.py`
  - generate `ml_a` (SDXL) and `ml_b` (PixArt-alpha) from `generation_prompts.csv`
  - supports `engine=diffusers` and `engine=stub` (deterministic test mode)
- `merge_covers_master.py`
  - merge `covers_master_real.csv` + `covers_master_ml_a.csv` + `covers_master_ml_b.csv`
  - validates source/group consistency and writes final `covers_master.csv`

## Expected Behavior

- Standardization is deterministic for the same input image.
- Manifest writers must preserve schema compatibility with README contracts.
- All output paths should be absolute/canonical when possible.

## Input Contract for Cover Standardization

`standardize-covers` expects a CSV with:
- `group_id`, `source`, `dataset`, `orig_id`, `caption_id`, `caption_text`
- `raw_image_path`, `qc_pass`, `qc_score`, `seed`

## ML Cover Generation Commands

```bash
# Real model generation (requires torch + diffusers)
python3 -m src.data.generate_ml_covers \
  --project-root . \
  --prompts-csv data/manifests/generation_prompts.csv \
  --engine diffusers

# Fast deterministic stub mode (for pipeline checks/tests)
python3 -m src.data.generate_ml_covers \
  --project-root . \
  --prompts-csv data/manifests/generation_prompts.csv \
  --engine stub
```

## Final Covers Merge Command

```bash
python3 -m src.data.merge_covers_master \
  --project-root . \
  --real-manifest data/manifests/covers_master_real.csv \
  --ml-a-manifest data/manifests/covers_master_ml_a.csv \
  --ml-b-manifest data/manifests/covers_master_ml_b.csv \
  --output-manifest data/manifests/covers_master.csv \
  --expected-groups 500
```
