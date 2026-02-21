# Cover Images

Unmodified carrier images used as input to the embedding pipelines.

## `real/` — 500 Real Photographs

| Source | Count | Notes |
|--------|-------|-------|
| RAISE | 250 | Demosaiced from RAW; stratified by scene category |
| COCO val2017 | 150 | Center-cropped; captions used as SD prompts |
| Flickr30k test | 100 | Center-cropped; captions used as SD prompts |

## `ml/` — 500 ML-Generated Images

| Source | Count | Notes |
|--------|-------|-------|
| Stable Diffusion v2.1 | 250 | Prompts from COCO/Flickr captions; seed=42 |
| StyleGAN3 | 250 | Seeds 0-249; truncation psi=0.7 |

## Quality Gate

All images must pass BRISQUE <= 50 (perceptual quality threshold). Images that fail are regenerated or resampled.

## Normalization

All images: 512x512 px, RGB, 8-bit, lossless PNG. RAISE images are demosaiced from RAW before normalization; all others are center-cropped and resized.
