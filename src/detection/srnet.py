from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class SRNetTrainingInput:
    """Closed-loop payload for the optional SRNet extension.

    Proposal alignment:
    - Reference specification: ``docs/proposals/proposal_updated_3.tex``,
      Section ``Chosen Approaches -> Steganalysis Detectors`` under
      ``Extension (time-permitting)``.

    Expected usage:
    - The primary study does not depend on this structure.
    - It exists only for the optional deep-detector extension reported
      separately from the confirmatory classical-detector analysis.
    """

    method: str
    fold: int
    x_train: Sequence[Any]
    y_train: Sequence[int]
    x_val: Sequence[Any]
    y_val: Sequence[int]
    random_seed: int = 42


@dataclass(frozen=True)
class SRNetModelArtifact:
    """Serializable in-memory SRNet artifact for optional extension runs."""

    method: str
    fold: int
    model_state: Any
    hyperparams: dict[str, Any]


def train_srnet_model(training_input: SRNetTrainingInput) -> SRNetModelArtifact:
    """Train the optional SRNet detector for one method/fold combination.

    Intended implementation:
    - Follow Boroumand, Chen, and Fridrich [boroumand2019srnet].
    - Restrict this work to the optional extension branch described in the
      proposal; the main pipeline must not depend on it.
    - Accept already prepared train/validation tensors or image batches in
      memory and return a serializable in-memory model artifact.
    """
    raise NotImplementedError("SRNet training is not implemented yet.")


def score_srnet_model(
    model: SRNetModelArtifact,
    x_samples: Sequence[Any],
) -> list[float]:
    """Score samples with a trained SRNet model.

    Intended implementation:
    - Consume the model artifact from ``train_srnet_model``.
    - Return one floating-point score per sample.
    - Keep all I/O outside this function so the runner remains the only layer
      that touches the filesystem.
    """
    raise NotImplementedError("SRNet inference is not implemented yet.")
