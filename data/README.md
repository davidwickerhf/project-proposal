# Data Directory

This directory stores all data artifacts for the image steganalysis pipeline.

Git policy:
- Image binaries are excluded from git.
- Only `.gitkeep` and `README.md` files are tracked in data folders.

## Structure

```
data/
├── covers/
│   ├── real/        <- Real cover images
│   ├── ml_a/        <- SDXL-generated cover images
│   └── ml_b/        <- PixArt-alpha-generated cover images
├── payloads/
│   ├── plain/{low,medium,high}/
│   └── encrypted/{low,medium,high}/
├── stego/
│   ├── lsb/{low,medium,high}/{plain,encrypted}/{real,ml_a,ml_b}/
│   └── dct/{low,medium,high}/{plain,encrypted}/{real,ml_a,ml_b}/
└── manifests/
    ├── covers_master.csv
    ├── payload_manifest.csv
    └── stego_manifest.csv
```

## Locked Cardinalities

- Groups (`group_id`): `500`
- Covers per group: `3` (`real`, `ml_a`, `ml_b`)
- Total covers: `1,500`
- Stego variants per cover: `2 methods x 3 payload levels x 2 encryption states = 12`
- Total stegos: `18,000`

## File Naming Convention

Cover image:

```
g{group_id:04d}__src-{source}.png
```

Stego image:

```
g{group_id:04d}__src-{source}__m-{method}__p-{payload}__e-{encryption}.png
```

Payload file:

```
g{group_id:04d}__p-{payload}__e-{encryption}.bin
```

## Image Format

All cover and stego images are normalized to:
- `512x512`
- `RGB`
- `8-bit`
- lossless PNG

## Notes

- Legacy folders such as `data/covers/ml/` reflect an older design and should not be used for new runs.
- The canonical layout can be created via:
  - `python3 -m src.pipeline.cli --project-root . init-layout`
