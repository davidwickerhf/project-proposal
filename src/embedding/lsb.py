from __future__ import annotations

from PIL import Image


def embed_lsb(
    cover_image: Image.Image,
    payload_bytes: bytes,
    payload_level: str,
    prng_key: int,
) -> Image.Image:
    """Apply LSB embedding to a cover image.

    Placeholder: implementation intentionally deferred.
    """
    raise NotImplementedError("LSB embedding is not implemented yet.")
