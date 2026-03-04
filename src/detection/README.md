# `src/detection` Guide

`detection/` contains trained and statistical detector interfaces.

## Files

- `srm.py`
  - `SRMTrainingInput` (in-memory train/val payload)
  - `SRMModelArtifact` (in-memory trained model payload)
  - `train_srm_ec_model(training_input) -> SRMModelArtifact`
  - `score_srm_ec_model(model, x_samples) -> list[float]`
- `statistical.py`
  - `rs_analysis_score(image: PIL.Image.Image) -> float`
  - `chi_square_score(image: PIL.Image.Image) -> float`
  - `block_dct_shift_score(image: PIL.Image.Image) -> float`

Closed-loop contract (all deferred detection functions):
- Inputs are in-memory artifacts only (feature matrices, labels, model object, images).
- Outputs are scores/model artifacts only.
- No direct file reads/writes inside detectors.
- Pipeline/orchestrator layers own all disk I/O.

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
