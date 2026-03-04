from __future__ import annotations

from PIL import Image


def embed_dct_qim(
    cover_image: Image.Image,
    payload_bytes: bytes,
    payload_level: str,
    delta: float,
) -> Image.Image:
    """Apply block-DCT QIM embedding to a cover image.

    Placeholder: implementation intentionally deferred.
    """
    raise NotImplementedError("DCT-QIM embedding is not implemented yet.")
