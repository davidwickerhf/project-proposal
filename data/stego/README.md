# Stego Images

Steganographic images produced by embedding payloads into cover images. Total: **12,000 images** (1,000 covers x 2 methods x 3 payloads x 2 encryption conditions).

## Directory Layout

```
stego/{method}/{payload_level}/{encryption}/{carrier_origin}/
```

- **method**: `lsb` (spatial-domain LSB substitution) or `dct` (frequency-domain DCT-QIM)
- **payload_level**: `low`, `medium`, or `high`
- **encryption**: `plain` (raw pseudorandom payload) or `encrypted` (AES-256-CBC pre-encrypted)
- **carrier_origin**: `real` or `ml`

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
- **encrypted/**: Same bitstream passed through AES-256-CBC encryption before embedding. Addresses RQ4 (encryption effect on detectability).

## Payload & Key

- Payload: pseudorandom bits generated with `numpy.random.default_rng(seed=42)`
- Embedding key (pixel selection PRNG): `key=12345`
- AES-256 key: 32-byte key derived from a fixed passphrase (for reproducibility)

## Pairing

Stego filenames match their cover filenames exactly, enabling direct cover-stego pairing for quality metrics (PSNR, SSIM, FSIM) and detection evaluation.
