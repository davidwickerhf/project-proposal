from __future__ import annotations

from PIL import Image


def embed_lsb(
    cover_image: Image.Image,
    payload_bytes: bytes,
    payload_level: str,
    prng_key: int,
) -> Image.Image:
    """Embed payload bytes into a cover image using an LSB strategy.

    Contract:
    - Input:
      - cover_image: in-memory image (expected canonical RGB 512x512 from pipeline).
      - payload_bytes: bytes to embed (already plain or encrypted upstream).
      - payload_level: one of {"low", "medium", "high"}.
      - prng_key: deterministic key for pseudo-random pixel/channel ordering.
    - Output:
      - stego image as a new in-memory ``PIL.Image.Image`` object.
    - Side effects:
      - none. This function must not read/write files.
    """
    raise NotImplementedError("LSB embedding is not implemented yet.")
