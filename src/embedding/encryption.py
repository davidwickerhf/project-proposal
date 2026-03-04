from __future__ import annotations


def encrypt_payload_aes_256_cbc(payload: bytes, key: bytes, iv: bytes) -> bytes:
    """Encrypt payload bytes using AES-256-CBC.

    Contract:
    - Input:
      - payload: arbitrary plaintext bytes to embed.
      - key: exactly 32 bytes (AES-256 key).
      - iv: exactly 16 bytes (CBC IV).
    - Output:
      - ciphertext bytes.
    - Side effects:
      - none. This function must not read/write files.
    """
    raise NotImplementedError("AES-256-CBC encryption is not implemented yet.")


def decrypt_payload_aes_256_cbc(ciphertext: bytes, key: bytes, iv: bytes) -> bytes:
    """Decrypt AES-256-CBC ciphertext back to plaintext bytes.

    Contract:
    - Input:
      - ciphertext: encrypted payload bytes.
      - key: exactly 32 bytes (AES-256 key).
      - iv: exactly 16 bytes (CBC IV used in encryption).
    - Output:
      - plaintext payload bytes.
    - Side effects:
      - none. This function must not read/write files.
    """
    raise NotImplementedError("AES-256-CBC decryption is not implemented yet.")
