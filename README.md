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

The implementation is designed to isolate the effect of **carrier source** on steganographic detectability while holding the rest of the pipeline fixed. The confirmatory scope is:
- `RQ1`: real vs pooled-ML source effect
- `RQ2`: generation-model effect within ML (ML-A vs ML-B)
- `RQ3`: payload interaction with source effects
- `RQ4`: embedding-method interaction with source
- `RQ5`: encryption interaction with source

## 2) Controlled Study Design

Primary design factors:
- Carrier source: `real`, `ML-A`, `ML-B`
- Embedding method: `LSB`, `DCT-QIM`
- Payload: `low`, `medium`, `high`
- Encryption condition: `plain`, `AES-256-CBC`

All preprocessing and evaluation settings are standardized to keep comparisons fair.

## 3) Dataset Construction

### Real carriers (500)
- `COCO`: 300 images
- `Flickr30k`: 200 images

### ML carriers (1,000)
- `ML-A (SDXL 1.0)`: 500 images
- `ML-B (PixArt-alpha)`: 500 images

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

Confirmatory detectors:
- `SRM+EC` (cross-method baseline for LSB and DCT)
- `RS Analysis` (LSB-specific confirmatory detector)
- `Block-DCT shift test` (DCT-specific confirmatory detector)

Sensitivity-only detector:
- `chi-square` attack for LSB robustness checks

## 7) Experiment Mapping

- `Exp1 (RQ1)`: compare real vs pooled-ML detectability under matched settings.
- `Exp2 (RQ2)`: compare ML-A vs ML-B detectability under matched settings.
- `Exp3 (RQ3)`: test source x payload interaction.
- `Exp4 (RQ4)`: test source x embedding-method interaction.
- `Exp5 (RQ5)`: test source x encryption interaction.

## 8) Detailed Validation Protocol (Working Notes)

### Detection metrics
- `ROC-AUC` (primary)
- Secondary summaries: accuracy at Youden's J, EER, and FPR at fixed FNR

### Quality control metrics
- `PSNR`, `SSIM`, `FSIM`
- `BRISQUE` used as generation-stage quality gate

### Detector usage rules
- LSB confirmatory: `SRM+EC`, `RS`
- DCT confirmatory: `SRM+EC`, `Block-DCT shift test`
- `chi-square` is reported as LSB sensitivity-only (not confirmatory)

### Statistical testing
- ANOVA on AUC with source, method, payload, and encryption factors
- Planned Wilcoxon signed-rank contrasts
- Effect sizes (`Cohen's d`) with uncertainty reporting
- Bonferroni correction across confirmatory comparisons

## 9) Planned Outputs

- Condition-level ROC curves and AUC tables
- Experiment-level summary plots (payload trends, encryption effects, interaction plots)
- Statistical result tables (p-values, corrected significance, effect sizes, confidence intervals)
- Quality metric tables and threshold checks

## 10) Implementation Sequence

1. Build/normalize datasets (real + ML)
2. Generate cover/stego sets for all conditions
3. Run SRM+EC, RS, block-DCT shift, and chi-square (sensitivity) scoring
4. Compute ROC/AUC and quality metrics
5. Run experiment-specific statistical analyses
6. Produce figures/tables for report and slides
