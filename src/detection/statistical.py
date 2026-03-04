from __future__ import annotations

from pathlib import Path


def rs_analysis_score(image_path: Path) -> float:
    """Return an RS-analysis detection score for one image.

    Placeholder: implementation intentionally deferred.
    """
    raise NotImplementedError("RS analysis is not implemented yet.")


def chi_square_score(image_path: Path) -> float:
    """Return a chi-square detection score for one image.

    Placeholder: implementation intentionally deferred.
    """
    raise NotImplementedError("Chi-square steganalysis is not implemented yet.")


def block_dct_shift_score(image_path: Path) -> float:
    """Return a block-DCT shift-test score for one image.

    Placeholder: implementation intentionally deferred.
    """
    raise NotImplementedError("Block-DCT shift test is not implemented yet.")
