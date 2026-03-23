# `src/evaluation` Guide

`evaluation/` contains split generation, metric aggregation, and plotting helpers.

## Locked Split Design

- total groups: `500`
- folds: `5`
- per fold:
  - train groups: `350`
  - validation groups: `50`
  - test groups: `100`

All three source variants for the same `group_id` stay in the same partition to prevent leakage.

## Outputs

- `results/splits/splits_grouped5fold.json`
- `results/metrics/*.csv`
- `results/figures/*.png`
