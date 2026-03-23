# `src/detection` Guide

`detection/` now mirrors the final proposal’s detector plan.

## Primary Detectors

- `rs_analysis_score(image)`
- `chi_square_spatial_score(image)`
- `sample_pairs_score(image)`
- `chi_square_dct_score(jpeg_bytes)`
- `calibration_chi_square_score(jpeg_bytes, jpeg_quality=95)`

All five are training-free and belong to the mainline experiment.

## Optional Extension

- `srnet.py`
  - `SRNetTrainingInput`
  - `SRNetModelArtifact`
  - `train_srnet_model(...)`
  - `score_srnet_model(...)`

SRNet is not part of the default pipeline. It exists only for the proposal’s time-permitting extension.
