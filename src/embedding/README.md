# `src/embedding` Guide

`embedding/` implements payload transformation and stego generation.

## Deferred Files

- `encryption.py`
  - `encrypt_payload_aes_256_cbc(payload, key, iv) -> bytes`
  - `decrypt_payload_aes_256_cbc(ciphertext, key, iv) -> bytes`
- `lsb.py`
  - `embed_lsb(cover_image, payload_bytes, payload_level, prng_key) -> Image`
- `dct.py`
  - `embed_dct_qim(cover_image, payload_bytes, payload_level, delta) -> Image`

## Implementation Requirements

- Keep output image in canonical format constraints (`RGB`, `512x512`, PNG save-safe).
- Respect payload-level policies (`low`, `medium`, `high`) from pipeline params.
- Use deterministic behavior under fixed seeds/keys.
- Do not change filename/path logic here; it is owned by `common/contracts.py`.

## Integration Notes

- Runner orchestrates method selection and manifest joins.
- `run-embedding-stage --execute` will call these functions directly.
