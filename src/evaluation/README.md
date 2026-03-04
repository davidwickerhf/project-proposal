# `src/evaluation` Guide

`evaluation/` contains experiment split and scoring utilities.

## Files

- `splits.py`
  - grouped 5-fold split generation by `group_id`
  - returns train/val/test group lists for each fold

## Locked Split Design

- Total groups: `500`
- Folds: `5`
- Per fold:
  - test groups: `100`
  - validation groups: `50`
  - training groups: `350`

## Why Grouped Splits

All source variants (`real`, `ml_a`, `ml_b`) for the same `group_id` must stay in the same partition to prevent leakage.

## Downstream Usage

- Pipeline writes split artifact: `results/splits/splits_grouped5fold.json`
- SRM training-job planner reads this split and expands per-method jobs.
