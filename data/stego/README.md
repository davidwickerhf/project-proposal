# Stego Images

Steganographic images produced by embedding payloads into cover images.

Total: **18,000 images**  
Computation: `1,500 covers x 2 methods x 3 payload levels x 2 encryption states`

## Directory Layout

```
stego/{method}/{payload_level}/{encryption}/{source}/
```

- **method**: `lsb` (spatial-domain LSB substitution) or `dct` (frequency-domain DCT-QIM)
- **payload_level**: `low`, `medium`, or `high`
- **encryption**: `plain` (raw pseudorandom payload) or `encrypted` (AES-256-CBC pre-encrypted)
- **source**: `real`, `ml_a`, or `ml_b`

## Embedding Methods

### LSB Substitution
Replaces the k least significant bits of PRNG-selected pixel channels.

| Level | k | Pixel fraction | Approx bpp |
|-------|---|----------------|------------|
| low | 1 | 25% | ~0.08 |
| medium | 1 | 50% | ~0.16 |
| high | 2 | 50% | ~0.32 |

### DCT-QIM
Embeds bits into mid-frequency DCT coefficients (zigzag positions 10-54) of 8x8 blocks via Quantization Index Modulation (delta=20.0).

| Level | Coefficient fraction | Approx bpp |
|-------|---------------------|------------|
| low | 10% | ~0.08 |
| medium | 25% | ~0.16 |
| high | 50% | ~0.32 |

## Encryption

- **plain/**: Payload is a pseudorandom bitstream (fixed-seed PRNG), embedded directly.
- **encrypted/**: Same bitstream passed through AES-256-CBC encryption before embedding. Addresses RQ5 (encryption interaction).

## Payload & Key

- Payload: pseudorandom bits generated with `numpy.random.default_rng(seed=42)`
- Embedding key (pixel selection PRNG): `key=12345`
- AES-256 key: 32-byte key derived from a fixed passphrase (for reproducibility)

## Pairing

Stego filenames are deterministic and self-describing:

```
g{group_id:04d}__src-{source}__m-{method}__p-{payload}__e-{encryption}.png
```

This supports direct join with cover files by `group_id + source` for:
- quality metrics (`PSNR`, `SSIM`, `FSIM`)
- detector evaluation
