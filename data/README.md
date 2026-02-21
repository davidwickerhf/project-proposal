# Data Directory

This directory stores all image data for the project. Images are excluded from git via `.gitignore` (only `.gitkeep` and `README.md` files are tracked).

## Structure

```
data/
├── covers/          <- Original unmodified images (1,000 total)
│   ├── real/        <- 500 real photographs (RAISE 250, COCO 150, Flickr30k 100)
│   └── ml/          <- 500 ML-generated images (SD v2.1 250, StyleGAN3 250)
└── stego/           <- Steganographic images (12,000 total)
    ├── lsb/         <- LSB-embedded images
    │   ├── low/     <- ~0.08 bpp (k=1, 25% pixels)
    │   ├── medium/  <- ~0.16 bpp (k=1, 50% pixels)
    │   └── high/    <- ~0.32 bpp (k=2, 50% pixels)
    └── dct/         <- DCT-QIM-embedded images
        ├── low/     <- ~0.08 bpp (10% mid-freq coefficients)
        ├── medium/  <- ~0.16 bpp (25% mid-freq coefficients)
        └── high/    <- ~0.32 bpp (50% mid-freq coefficients)
```

Each payload level directory contains:
```
{low,medium,high}/
├── plain/           <- Unencrypted payload
│   ├── real/        <- Stego images from real covers
│   └── ml/          <- Stego images from ML-generated covers
└── encrypted/       <- AES-256-CBC encrypted payload
    ├── real/
    └── ml/
```

## File Naming Convention

All images follow the format:

```
{source}_{id:04d}_{category}.png
```

- `source`: `raise`, `coco`, `flickr`, `sd21`, or `sg3`
- `id`: zero-padded 4-digit integer
- `category`: `outdoor`, `indoor`, `portrait`, `macro`, or `other`

Examples: `raise_0001_outdoor.png`, `sd21_0042_portrait.png`

Stego images share the same filename as their cover, so pairing is trivial.

## Image Format

All images are normalized to: **512x512 px, RGB, 8-bit depth, lossless PNG**.
