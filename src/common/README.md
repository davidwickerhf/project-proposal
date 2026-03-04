# `src/common` Guide

`common/` defines shared contracts used by every stage.

## Key File

- `contracts.py`

## What It Owns

- Allowed categorical values:
  - `source in {real, ml_a, ml_b}`
  - `method in {lsb, dct}`
  - `payload_level in {low, medium, high}`
  - `encryption in {plain, encrypted}`
- Filename constructors for covers, payloads, and stegos.
- Canonical path builder (`PipelinePaths`) for `data/` and `results/`.
- Layout creation (`ensure_layout`).

## Implementation Rule

Do not duplicate naming/path logic in other modules. Import and reuse these helpers to avoid drift between stages.
