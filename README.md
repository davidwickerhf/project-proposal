# Does the Source of Carrier Image Affect Steganographic Detectability?

Comparative study of steganographic detectability on real vs ML-generated image carriers.

**Course:** Project 2.2, Department of Advanced Computing Sciences, Maastricht University  
**Team:** Abdul Moiz Akbar, Malo Coquin, Daria Gjonbalaj, Nico Muller-Spath, Jimena Narvaez del Cid, David Wicker, Nikolas Zouros  
**Date:** February 2026

## Quick Links

- Midway proposal (final, LaTeX): `docs/proposals/midway_proposal_final.tex`
- Midway proposal (final, PDF): `docs/proposals/midway_proposal_final.pdf`
- Midway proposal slides (full): `docs/slides/midway_proposal_slides.tex`
- Midway proposal slides (full PDF): `docs/slides/midway_proposal_slides.pdf`
- Midway proposal slides (5 min): `docs/slides/midway_proposal_slides_5min.tex`
- Midway proposal slides (5 min PDF): `docs/slides/midway_proposal_slides_5min.pdf`

## Repository Structure

```text
project-proposal/
├── README.md
├── docs/
│   ├── proposals/
│   │   ├── midway_proposal_final.tex
│   │   └── midway_proposal_final.pdf
│   └── slides/
│       ├── midway_proposal_slides.tex
│       ├── midway_proposal_slides.pdf
│       ├── midway_proposal_slides_5min.tex
│       └── midway_proposal_slides_5min.pdf
├── src/
│   ├── data/
│   ├── embedding/
│   ├── detection/
│   └── evaluation/
├── data/
├── results/
└── notebooks/
```

## Examiner Comments and Proposed Responses

### 1) Reframe image-quality research question

**Examiner comment:** avoid user studies if possible; rely on standardized quality metrics.

**Proposed response:** keep `RQ1-RQ4` as core confirmatory questions, and frame `RQ5` as an additional technical quality/validity check:
- Evaluate quality with `PSNR`, `SSIM`, `FSIM` (and optional no-reference metrics such as `BRISQUE`).
- Include carrier source in quality analysis (real vs ML).
- Use quality metrics as a confound check for detectability differences, not as a separate user-study endpoint.

### 2) Consider LLM-based quality/source testing

**Examiner idea:** test image quality and carrier source with an LLM.

**Proposed response:** keep LLM/VLM-based judgment optional and exploratory only (appendix-level), not primary evidence. Primary evidence remains objective metrics and statistical testing.

### 3) Compare different encryption models if feasible

**Examiner comment:** compare encryption variants only if scope remains manageable.

**Proposed response:** keep `plain vs AES-256-CBC` as the core confirmatory comparison. If time permits, add one reduced-scope robustness check (for example, `AES-CTR` or `ChaCha20` on a subset).

## Technical Implementation Plan

## 1) Objective and Scope

The implementation is designed to isolate the effect of **carrier origin** on steganographic detectability while holding the rest of the pipeline fixed. The confirmatory scope is:
- `RQ1`: carrier-origin effect
- `RQ2`: payload trend effect
- `RQ3`: encryption effect and interaction with origin
- `RQ4`: embedding-method interaction with origin

`RQ5` (image quality) is treated as an additional quality/validity check.

## 2) Controlled Study Design

Primary design factors:
- Carrier origin: `real`, `ML-generated`
- Embedding method: `LSB`, `DCT-QIM`
- Payload: `low`, `medium`, `high`
- Encryption condition: `plain`, `AES-256-CBC`

All preprocessing and evaluation settings are standardized to keep comparisons fair.

## 3) Dataset Construction

### Real carriers (500)
- `RAISE`: 250 images
- `COCO`: 150 images
- `Flickr30k`: 100 images

### ML carriers (500)
- `Stable Diffusion v2.1`: 250 images
- `StyleGAN3`: 250 images

### Standardization
- Resize/normalize to `512x512`, RGB, 8-bit PNG.
- Apply quality filtering for generated images (for example BRISQUE threshold).

## 4) Embedding Pipeline

### Spatial-domain method: LSB
- PRNG-keyed pixel selection.
- `k=1` for low/medium payload, `k=2` for high payload.

### Frequency-domain method: DCT-QIM
- `8x8` block DCT.
- Mid-frequency coefficient embedding with QIM.

### Encryption branch
- Optional pre-embedding encryption with `AES-256-CBC`.

## 5) Cover/Stego Pair Construction (AUC Requirement)

For every condition, build paired classes:
- **Negative class:** cover images (no message)
- **Positive class:** stego images (embedded message)

This ensures valid ROC/AUC computation per condition.

## 6) Steganalysis Detectors

Main detectors (analyzed separately):
- `RS Analysis` (Regular-Singular Analysis)
- `SRM+FLD` (Spatial Rich Model + Fisher Linear Discriminant ensemble)

Supplementary check:
- `chi-square` attack for LSB-oriented diagnostics

## 7) Experiment Mapping

- `Exp1 (RQ1)`: compare real vs ML AUC under matched settings; detector-specific tests (`H1`, `H2`).
- `Exp2 (RQ2)`: test payload trend of real-vs-ML AUC gap (`H3`).
- `Exp3 (RQ3)`: compare plain vs AES and test encryption-by-origin interaction (`H4`, `H5`).
- `Exp4 (RQ4)`: test carrier-origin x embedding-method interaction (`H6`).
- `Exp5 (RQ5 additional)`: evaluate quality changes across method, payload, and carrier source.

## 8) Metrics and Statistical Validation

### Detectability (primary)
- `ROC-AUC`
- Optional secondary summaries: accuracy at Youden's J, EER, constrained FPR/FNR summaries

### Quality (additional)
- `PSNR`, `SSIM`, `FSIM`

### Statistics
- Wilcoxon signed-rank for paired condition comparisons
- Two-way ANOVA interaction testing where appropriate
- Effect sizes (Cohen's d) and uncertainty intervals
- Bonferroni correction across 6 confirmatory hypotheses:
  - `alpha_adj = 0.05 / 6 approx 0.0083`

## 9) Planned Outputs

- Condition-level ROC curves and AUC tables
- Experiment-level summary plots (payload trends, encryption effects, interaction plots)
- Statistical result tables (p-values, corrected significance, effect sizes, confidence intervals)
- Quality metric tables and threshold checks

## 10) Implementation Sequence

1. Build/normalize datasets (real + ML)
2. Generate cover/stego sets for all conditions
3. Run RS and SRM+FLD scoring
4. Compute ROC/AUC and quality metrics
5. Run experiment-specific statistical analyses
6. Produce figures/tables for report and slides
