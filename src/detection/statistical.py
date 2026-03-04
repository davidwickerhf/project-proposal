from __future__ import annotations

from PIL import Image


def rs_analysis_score(image: Image.Image) -> float:
    """Return an RS-analysis score for one in-memory image.

    Contract:
    - Input:
      - image: in-memory carrier/stego image.
    - Output:
      - scalar score where larger values indicate stronger LSB-stego evidence.
    - Side effects:
      - none. This function must not read/write files.
    """
    raise NotImplementedError("RS analysis is not implemented yet.")


def chi_square_score(image: Image.Image) -> float:
    """Return a chi-square score for one in-memory image.

    Contract:
    - Input:
      - image: in-memory carrier/stego image.
    - Output:
      - scalar score where larger values indicate stronger LSB-stego evidence.
    - Side effects:
      - none. This function must not read/write files.
    """
    raise NotImplementedError("Chi-square steganalysis is not implemented yet.")


def block_dct_shift_score(image: Image.Image) -> float:
    """Return a block-DCT shift-test score for one in-memory image.

    Contract:
    - Input:
      - image: in-memory carrier/stego image.
    - Output:
      - scalar score where larger values indicate stronger DCT-domain stego evidence.
    - Side effects:
      - none. This function must not read/write files.
    """
    raise NotImplementedError("Block-DCT shift test is not implemented yet.")
