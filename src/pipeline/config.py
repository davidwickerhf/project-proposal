from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.common.contracts import PipelinePaths


@dataclass(frozen=True)
class PipelineConfig:
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
        return PipelinePaths.from_project_root(self.project_root)

    @classmethod
    def from_project_root(cls, project_root: Path) -> "PipelineConfig":
        return cls(project_root=project_root.resolve())
