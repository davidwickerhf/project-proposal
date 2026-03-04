from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SRMRunSpec:
    fold: int
    method: str
    train_groups: int
    val_groups: int
    test_groups: int
    train_samples: int
    val_samples: int
    test_samples: int


def train_srm_ec_model(method: str, fold: int) -> None:
    """Train SRM+EC for a given method/fold.

    Placeholder for SRM feature extraction and EC fitting.
    """
    raise NotImplementedError("SRM+EC training is not implemented yet.")


def score_srm_ec_model(method: str, fold: int) -> None:
    """Score SRM+EC predictions for a given method/fold.

    Placeholder for model inference.
    """
    raise NotImplementedError("SRM+EC inference is not implemented yet.")
