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

## Expected Behavior

- Standardization is deterministic for the same input image.
- Manifest writers must preserve schema compatibility with README contracts.
- All output paths should be absolute/canonical when possible.

## Input Contract for Cover Standardization

`standardize-covers` expects a CSV with:
- `group_id`, `source`, `dataset`, `orig_id`, `caption_id`, `caption_text`
- `raw_image_path`, `qc_pass`, `qc_score`, `seed`
