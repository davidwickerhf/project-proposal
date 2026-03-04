from __future__ import annotations

"""Pipeline configuration and experiment defaults.

This module defines the frozen runtime configuration for all pipeline stages.
Values here intentionally encode the locked experimental design from README:
- 500 groups
- 5 folds
- 3 payload levels
- deterministic seeds
"""

from dataclasses import dataclass, field
from pathlib import Path

from src.common.contracts import PipelinePaths


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable configuration for the full data/embedding/detection pipeline.

    Notes:
    - ``project_root`` is the base for all relative manifest paths.
    - Seed values are intentionally centralized so re-runs remain reproducible.
    - Payload sizes are in bits, keyed by payload level.
    """

    project_root: Path
    image_size: tuple[int, int] = (512, 512)
    n_groups: int = 500
    n_folds: int = 5
    split_seed: int = 42
    payload_seed: int = 42
    embed_seed: int = 42
    lsb_prng_key: int = 12345
    aes_key_id: str = "aes256cbc-v1"
    payload_bits_by_level: dict[str, int] = field(
        default_factory=lambda: {"low": 32768, "medium": 65536, "high": 131072}
    )
    dct_delta: float = 20.0

    @property
    def paths(self) -> PipelinePaths:
        """Return canonical repository paths derived from ``project_root``."""
        return PipelinePaths.from_project_root(self.project_root)

    @classmethod
    def from_project_root(cls, project_root: Path) -> "PipelineConfig":
        """Build config with defaults and a resolved absolute project root."""
        return cls(project_root=project_root.resolve())
