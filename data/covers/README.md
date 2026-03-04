# Cover Images

Unmodified carrier images used as input to the embedding pipelines.

## Source Directories

- `real/`: real photographic covers from curated COCO/Flickr30k set
- `ml_a/`: SDXL-generated covers
- `ml_b/`: PixArt-alpha-generated covers

## Locked Counts

- Total groups (`group_id`): `500`
- Covers per group: `3` (`real`, `ml_a`, `ml_b`)
- Total covers: `1,500`
- Per source: `500`

## Quality Gate

- All covers must pass the generation/selection quality gate before freeze.
- Store QC outcomes in `data/manifests/covers_master.csv` (`qc_pass`, `qc_score`).

## Normalization

All covers are standardized to:
- `512x512`
- `RGB`
- `8-bit`
- lossless PNG

Standardization is performed by:
- `python3 -m src.pipeline.cli --project-root . standardize-covers --input-index <raw_cover_index.csv>`

## Naming

Cover naming must follow:

```
g{group_id:04d}__src-{source}.png
```

where `source in {real, ml_a, ml_b}`.
