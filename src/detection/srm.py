from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


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


@dataclass(frozen=True)
class SRMTrainingInput:
    """Closed-loop training payload for one method/fold run."""

    method: str
    fold: int
    x_train: Sequence[Sequence[float]]
    y_train: Sequence[int]
    x_val: Sequence[Sequence[float]]
    y_val: Sequence[int]
    random_seed: int = 42


@dataclass(frozen=True)
class SRMModelArtifact:
    """Serializable in-memory model artifact returned by training."""

    method: str
    fold: int
    model_state: Any
    hyperparams: dict[str, Any]


def train_srm_ec_model(training_input: SRMTrainingInput) -> SRMModelArtifact:
    """Train SRM+EC for one method/fold in-memory dataset.

    Contract:
    - Input:
      - SRMTrainingInput with train/val features + labels for one method and fold.
    - Output:
      - SRMModelArtifact containing model state and hyperparameters in memory.
    - Side effects:
      - none. This function must not read/write files.
    """
    raise NotImplementedError("SRM+EC training is not implemented yet.")


def score_srm_ec_model(
    model: SRMModelArtifact,
    x_samples: Sequence[Sequence[float]],
) -> list[float]:
    """Score samples with a trained SRM+EC model.

    Contract:
    - Input:
      - model: in-memory SRMModelArtifact from ``train_srm_ec_model``.
      - x_samples: feature rows to score.
    - Output:
      - list of detector scores/probabilities, one per input row.
    - Side effects:
      - none. This function must not read/write files.
    """
    raise NotImplementedError("SRM+EC inference is not implemented yet.")
