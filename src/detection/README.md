# `src/detection` Guide

`detection/` contains trained and statistical detector interfaces.

## Files

- `srm.py`
  - SRM+EC training and scoring entrypoints
  - per-method, per-fold model execution
- `statistical.py`
  - `rs_analysis_score(image_path) -> float`
  - `chi_square_score(image_path) -> float`
  - `block_dct_shift_score(image_path) -> float`

## Detector Policy

- Confirmatory:
  - SRM+EC (trained)
  - RS (LSB)
  - Block-DCT shift (DCT)
- Sensitivity-only:
  - chi-square (LSB)

## Training Scope

- SRM is trained per method (`lsb`, `dct`) and per fold.
- Total expected SRM runs for full experiment: `2 methods x 5 folds = 10`.

## Output Expectation

All detector outputs should be exportable into prediction tables with:
- `fold`, `detector`, `group_id`, `source`, `method`, `payload_level`, `encryption`, `label`, `score`
