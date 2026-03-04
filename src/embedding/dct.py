from __future__ import annotations

from PIL import Image


def embed_dct_qim(
    cover_image: Image.Image,
    payload_bytes: bytes,
    payload_level: str,
    delta: float,
) -> Image.Image:
    """Embed payload bytes using a block-DCT QIM approach.

    Contract:
    - Input:
      - cover_image: in-memory image (expected canonical RGB 512x512 from pipeline).
      - payload_bytes: bytes to embed (already plain or encrypted upstream).
      - payload_level: one of {"low", "medium", "high"}.
      - delta: positive QIM quantization step.
    - Output:
      - stego image as a new in-memory ``PIL.Image.Image`` object.
    - Side effects:
      - none. This function must not read/write files.
    """
    raise NotImplementedError("DCT-QIM embedding is not implemented yet.")
