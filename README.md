# Does the Source of Carrier Image Affect Steganographic Detectability?

> A comparative study of real vs. ML-generated image steganography

**Course:** Project 2.2, Department of Advanced Computing Sciences, Maastricht University
**Team:** Abdul Moiz Akbar, Malo Coquin, Daria Gjonbalaj, Nico Müller-Späth, Jimena Naravaez del Cid, David Wicker, Nikolas Zouros
**Date:** February 2026

## Table of Contents

1. [Project Overview](#project-overview)
2. [Midway Proposal](#midway-proposal)
   - [Motivation and Problem Statement](#motivation-and-problem-statement)
   - [Research Questions](#research-questions)
   - [Chosen Approaches](#chosen-approaches)
   - [Experiments](#experiments)
   - [Prototype](#prototype)
   - [Related Work](#related-work)
   - [Relation to Curriculum](#relation-to-curriculum)
   - [Planning](#planning)
   - [Minimal Passing Requirements](#minimal-passing-requirements)
   - [References](#references)
3. [Implementation](#implementation)
   - [1. Overview](#1-overview)
   - [2. Dataset Construction](#2-dataset-construction-phase-1)
   - [3. Steganographic Embedding](#3-steganographic-embedding-phase-2)
   - [4. Steganalysis Detectors](#4-steganalysis-detectors-phase-3)
   - [5. Experimental Conditions](#5-experimental-conditions-phase-4)
   - [6. Evaluation Metrics](#6-evaluation-metrics-phase-5)
   - [7. Expected Results Structure](#7-expected-results-structure)
   - [8. Division of Labor & Timeline](#8-division-of-labor--timeline)
   - [9. References](#9-references)

---

## Project Overview

This project investigates whether the **origin of a carrier image** (real photograph vs. ML-generated) affects the detectability of steganographic content. We use a controlled factorial experiment with 1,000 images (500 real + 500 ML-generated), two embedding methods (LSB and DCT), three payload levels, and two steganalysis detectors (RS Analysis and SRM+FLD).

### Repository Structure

```
├── README.md                   ← This file: methodology & implementation plan
├── docs/
│   ├── proposals/              ← Research & midway proposals (LaTeX + PDF)
│   ├── slides/                 ← Presentation slides (LaTeX + PDF)
│   └── references/             ← Reference review notes
├── src/
│   ├── data/                   ← Dataset construction scripts
│   ├── embedding/              ← LSB, DCT, AES embedding pipelines
│   ├── detection/              ← RS Analysis, chi-square, SRM+FLD
│   └── evaluation/             ← Metrics computation & statistical analysis
├── data/
│   ├── covers/                 ← Original images (real/ and ml/)
│   └── stego/                  ← Embedded images (lsb|dct / low|medium|high / plain|encrypted / real|ml)
├── results/
│   ├── metrics/                ← CSV/JSON detection & quality metrics
│   ├── figures/                ← ROC curves, heatmaps, bar charts
│   └── tables/                 ← ANOVA tables, summary statistics
└── notebooks/                  ← Jupyter analysis notebooks
```

---

## Midway Proposal

### Motivation and Problem Statement

[Image steganography](https://en.wikipedia.org/wiki/Steganography) (the practice of hiding secret information within digital images) is a fundamental topic in information security ([Petitcolas et al., 1999](https://doi.org/10.1109/5.771065); [Cheddad et al., 2010](https://doi.org/10.1016/j.sigpro.2009.08.010)). The detectability of hidden content is not absolute: it depends critically on the *statistical properties of the carrier image*. [Steganalysis](https://en.wikipedia.org/wiki/Steganalysis) systems exploit the subtle distributional changes that embedding introduces into pixel values, and a well-chosen carrier that already exhibits high local variability can mask these changes ([Hussain et al., 2018](https://doi.org/10.1016/j.image.2018.03.012); [Fridrich & Kodovsky, 2012](https://doi.org/10.1109/TIFS.2012.2190402)). The implicit assumption underlying nearly all published steganalysis work is that carriers are *real photographs*: images captured by digital cameras with well-characterized noise distributions, sensor patterns, and compression histories.

This assumption is increasingly untenable. Generative AI models including [Stable Diffusion](https://en.wikipedia.org/wiki/Stable_Diffusion) ([Rombach et al., 2022](https://arxiv.org/abs/2112.10752)), [StyleGAN3](https://en.wikipedia.org/wiki/StyleGAN) ([Karras et al., 2021](https://proceedings.neurips.cc/paper/2021/hash/076ccd93ad68be51f23707988e934906-Abstract.html)), [DALL-E 3](https://en.wikipedia.org/wiki/DALL-E), and [Midjourney](https://en.wikipedia.org/wiki/Midjourney) now produce images that are perceptually indistinguishable from real photographs, yet are synthesized through entirely different processes: [latent diffusion](https://en.wikipedia.org/wiki/Diffusion_model) over learned image manifolds, [adversarial training](https://en.wikipedia.org/wiki/Generative_adversarial_network) against a discriminator, or transformer-based token prediction. Each process imposes a *distinct statistical fingerprint* on the output: diffusion models produce characteristic noise residual patterns, [GANs](https://en.wikipedia.org/wiki/Generative_adversarial_network) leave spectral artifacts at high frequencies, and all generative models operate within learned data distributions that differ from the empirical distribution of camera photographs ([Wang et al., 2020](https://arxiv.org/abs/1912.11035); [Corvi et al., 2023](https://arxiv.org/abs/2211.00680)).

The central problem this study addresses is: **do existing steganalysis methods, designed and validated on photographs, remain effective when the carrier image is ML-generated?** This question has three concrete implications:

- **Security:** If ML-generated images are harder to steganalyze, adversaries could trivially evade current detectors by switching to synthetic carriers, without changing the embedding algorithm at all.
- **Scientific:** The interaction between a generative model's learned data distribution and the distributional perturbation caused by steganographic embedding is theoretically unexplored. Understanding it requires controlled empirical study.
- **Practical:** As AI-generated images proliferate across social media and digital communications, the steganographic attack surface is silently expanding. Practitioners need evidence on whether existing tools require retraining or adaptation.

Prior work most closely related to ours, [De et al. (2022)](https://doi.org/10.1186/s13638-022-02190-8), showed that AI-generated images can achieve statistically undetectable steganographic embedding using [minimum-entropy coupling](https://en.wikipedia.org/wiki/Coupling_(probability)). However, that study used a bespoke probabilistic embedding scheme and did not systematically compare standardized methods ([LSB](https://en.wikipedia.org/wiki/Least_significant_bit), [DCT](https://en.wikipedia.org/wiki/Discrete_cosine_transform)) across real vs. ML-generated carriers. No existing work provides a controlled comparison of carrier origin under identical embedding and steganalysis conditions.

We propose to fill this gap with a [factorial experiment](https://en.wikipedia.org/wiki/Factorial_experiment) (2 x 2 x 3 x 2): two carrier types (real photographs vs. ML-generated), two embedding methods (spatial [LSB](https://en.wikipedia.org/wiki/Least_significant_bit) and frequency-domain [DCT](https://en.wikipedia.org/wiki/Discrete_cosine_transform)), three payload levels, and two steganalysis detectors ([RS Analysis](https://en.wikipedia.org/wiki/Steganalysis#RS_analysis) and [SRM](https://ieeexplore.ieee.org/document/6197267)+[FLD](https://en.wikipedia.org/wiki/Linear_discriminant_analysis#Fisher's_linear_discriminant)). The study uses 500 real and 500 ML-generated images and is designed to be completed within 7 weeks using only CPU-based open-source tools.

### Research Questions

- **RQ1 (Carrier Origin Effect):** Does the origin of a carrier image (real photograph versus ML-generated) affect how easily hidden data can be detected?
  *Specifically:* Within a 7-week study using 500 real and 500 ML-generated images embedded with identical [LSB](https://en.wikipedia.org/wiki/Least_significant_bit) and [DCT](https://en.wikipedia.org/wiki/Discrete_cosine_transform) methods at three payload levels, does carrier origin produce a statistically significant difference (alpha = 0.05, [Bonferroni-corrected](https://en.wikipedia.org/wiki/Bonferroni_correction)) in steganalysis [AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve), as measured by [RS Analysis](https://en.wikipedia.org/wiki/Steganalysis#RS_analysis) and [SRM](https://ieeexplore.ieee.org/document/6197267)+[FLD](https://en.wikipedia.org/wiki/Linear_discriminant_analysis#Fisher's_linear_discriminant)?

- **RQ2 (Payload Sensitivity):** Does increasing the amount of hidden data widen the detectability gap between real and ML-generated carriers?
  *Specifically:* Across Low (≈0.08 [bpp](https://en.wikipedia.org/wiki/Bits_per_pixel)), Medium (≈0.16 bpp), and High (≈0.32 bpp) payload rates, does the [AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve) gap between real and ML-generated carriers increase monotonically, and does this trend differ between [LSB](https://en.wikipedia.org/wiki/Least_significant_bit) and [DCT](https://en.wikipedia.org/wiki/Discrete_cosine_transform) embedding (interaction effect in two-way [ANOVA](https://en.wikipedia.org/wiki/Analysis_of_variance))?

- **RQ3 (Embedding Method Interaction):** Does the choice of embedding method (spatial-domain [LSB](https://en.wikipedia.org/wiki/Least_significant_bit) versus frequency-domain [DCT](https://en.wikipedia.org/wiki/Discrete_cosine_transform)) change how carrier origin influences detectability?
  *Specifically:* Does the embedding method (spatial [LSB](https://en.wikipedia.org/wiki/Least_significant_bit) vs. frequency-domain [DCT](https://en.wikipedia.org/wiki/Discrete_cosine_transform)) interact significantly with carrier origin in determining detectability, as quantified by a two-way [ANOVA](https://en.wikipedia.org/wiki/Analysis_of_variance) [F-statistic](https://en.wikipedia.org/wiki/F-test) with [Bonferroni correction](https://en.wikipedia.org/wiki/Bonferroni_correction) applied across six tested hypotheses?

- **RQ4 (Payload Encryption Effect):** Does encrypting the hidden payload before embedding make steganography harder or easier to detect?
  *Specifically:* When the embedded payload is pre-encrypted with [AES-256-CBC](https://en.wikipedia.org/wiki/Advanced_Encryption_Standard) before [LSB](https://en.wikipedia.org/wiki/Least_significant_bit) or [DCT](https://en.wikipedia.org/wiki/Discrete_cosine_transform) embedding, does the steganalysis [AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve) change significantly compared to unencrypted embedding, and does this effect differ between carrier types (real vs. ML-generated)?

### Chosen Approaches

#### Datasets

**Real images (500 total).** We draw from three established photographic datasets chosen for their diversity and research accessibility: **[RAISE](http://loki.disi.unitn.it/RAISE/)** ([Dang-Nguyen et al., 2015](https://doi.org/10.1145/2713168.2713194)) contributes 250 RAW-demosaiced [DSLR](https://en.wikipedia.org/wiki/Digital_single-lens_reflex_camera) images spanning outdoor, indoor, portrait, and macro scenes; **[COCO](https://cocodataset.org/)** ([Lin et al., 2014](https://arxiv.org/abs/1405.0312)) contributes 150 images from its validation split; and **[Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/)** ([Young et al., 2014](https://doi.org/10.1162/tacl_a_00166)) contributes 100 images. All images are normalized to 512 x 512 px, [RGB](https://en.wikipedia.org/wiki/RGB_color_model), 8-bit, lossless [PNG](https://en.wikipedia.org/wiki/PNG). RAISE is preferred as the primary source because its [RAW format](https://en.wikipedia.org/wiki/Raw_image_format) preserves camera sensor noise structure, which is the natural image statistics that steganalysis exploits.

**ML-generated images (500 total).** We generate two matched sets of 250 images each using **[Stable Diffusion v2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1)** ([Rombach et al., 2022](https://arxiv.org/abs/2112.10752)) (via the [`diffusers`](https://huggingface.co/docs/diffusers) library on Apple MPS) and **[StyleGAN3](https://github.com/NVlabs/stylegan3)** ([Karras et al., 2021](https://proceedings.neurips.cc/paper/2021/hash/076ccd93ad68be51f23707988e934906-Abstract.html)) (official NVIDIA PyTorch implementation). Prompts for SD are derived directly from COCO/Flickr30k captions to achieve semantic alignment with real images. A [BRISQUE](https://en.wikipedia.org/wiki/BRISQUE) ≤ 50 quality gate rejects perceptually degraded outputs.

These two generative paradigms ([latent diffusion](https://en.wikipedia.org/wiki/Diffusion_model) and [GAN](https://en.wikipedia.org/wiki/Generative_adversarial_network)-based synthesis) represent the dominant architectures in open-source image generation and are expected to impose distinct statistical signatures on the output.

#### Embedding Methods

We implement two canonical [steganographic](https://en.wikipedia.org/wiki/Steganography) methods spanning the two principal domains:

**[LSB](https://en.wikipedia.org/wiki/Least_significant_bit) substitution (spatial domain)** replaces the *k* least significant bits of each pixel channel value with pseudorandom message bits, using a [PRNG](https://en.wikipedia.org/wiki/Pseudorandom_number_generator)-keyed pixel selection mask. We test k = 1 (Low, Medium payload) and k = 2 (High payload). Payloads are optionally pre-encrypted with [AES-256-CBC](https://en.wikipedia.org/wiki/Advanced_Encryption_Standard) before embedding, addressing RQ1's encryption sub-question.

**[DCT](https://en.wikipedia.org/wiki/Discrete_cosine_transform)-based embedding (frequency domain)** partitions each image channel into non-overlapping 8 x 8 blocks, computes the 2D [DCT](https://en.wikipedia.org/wiki/Discrete_cosine_transform), and embeds bits into mid-frequency coefficients ([zigzag](https://en.wikipedia.org/wiki/JPEG#Entropy_coding) positions 10-54) via **[Quantization Index Modulation](https://en.wikipedia.org/wiki/Quantization_index_modulation) (QIM)** ([Chen & Wornell, 2001](https://doi.org/10.1109/18.923725)): C'_i = Delta * round(C_i / Delta) +/- Delta/4, where the sign encodes the message bit. [DCT](https://en.wikipedia.org/wiki/Discrete_cosine_transform) embedding is chosen alongside [LSB](https://en.wikipedia.org/wiki/Least_significant_bit) to test whether frequency-domain methods are more sensitive to carrier origin (H3), since DCT coefficients reflect the generative model's learned spectral distribution directly.

#### Steganalysis Detectors

Our detector selection is deliberately scoped to *classical signal processing and statistics*, consistent with this project's cryptography/steganography focus:

| Detector | Type | Training | Domain | Time |
|---|---|---|---|---|
| [RS Analysis](https://en.wikipedia.org/wiki/Steganalysis#RS_analysis) ([Fridrich et al., 2001](https://doi.org/10.1109/93.959097)) | Statistical | None | Any | ~2 s/img |
| [Chi-square attack](https://en.wikipedia.org/wiki/Chi-squared_test) ([Westfeld & Pfitzmann, 1999](https://doi.org/10.1007/10719724_5)) | Statistical | None | [LSB](https://en.wikipedia.org/wiki/Least_significant_bit) | <1 s/img |
| [SRM](https://ieeexplore.ieee.org/document/6197267)+[FLD](https://en.wikipedia.org/wiki/Linear_discriminant_analysis#Fisher's_linear_discriminant) ([Fridrich & Kodovsky, 2012](https://doi.org/10.1109/TIFS.2012.2190402)) | Classical ML | Labeled | LSB+DCT | <30 min |

**[RS Analysis](https://en.wikipedia.org/wiki/Steganalysis#RS_analysis)** ([Fridrich et al., 2001](https://doi.org/10.1109/93.959097)) partitions an image into pixel groups and classifies each as Regular or Singular by a smoothness function; [LSB](https://en.wikipedia.org/wiki/Least_significant_bit) embedding shifts the R/S ratio predictably, yielding an analytical estimate of embedding rate. Since it requires no training, any detection difference between real and ML-generated carriers is attributable solely to carrier statistics and not to classifier bias.

**[SRM](https://ieeexplore.ieee.org/document/6197267) + [Fisher Linear Discriminant (FLD)](https://en.wikipedia.org/wiki/Linear_discriminant_analysis#Fisher's_linear_discriminant) ensemble** ([Fridrich & Kodovsky, 2012](https://doi.org/10.1109/TIFS.2012.2190402)) extracts ~35,000-dimensional [co-occurrence](https://en.wikipedia.org/wiki/Co-occurrence_matrix) feature vectors from high-pass residuals, then classifies with an ensemble of [FLD](https://en.wikipedia.org/wiki/Linear_discriminant_analysis#Fisher's_linear_discriminant) classifiers. It handles [DCT](https://en.wikipedia.org/wiki/Discrete_cosine_transform) embedding better than the training-free methods and its hand-crafted features are hypothesised to generalise better across the real/ML boundary than learned neural network representations. Implemented with [scikit-learn](https://scikit-learn.org/)'s `SGDClassifier`; 3-fold [stratified CV](https://en.wikipedia.org/wiki/Cross-validation_(statistics)#Stratified_k-fold_cross-validation).

The [chi-square attack](https://en.wikipedia.org/wiki/Chi-squared_test) ([Westfeld & Pfitzmann, 1999](https://doi.org/10.1007/10719724_5)) is applied as a supplementary check on [LSB](https://en.wikipedia.org/wiki/Least_significant_bit) results.

#### Validation

**Detection:** [ROC-AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve) (primary, threshold-independent); accuracy at [Youden's J](https://en.wikipedia.org/wiki/Youden%27s_J_statistic); [EER](https://en.wikipedia.org/wiki/Biometrics#Performance); [FPR](https://en.wikipedia.org/wiki/False_positive_rate) at 5% [FNR](https://en.wikipedia.org/wiki/False_positives_and_false_negatives).
**Image quality:** [PSNR](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio) (>40 dB target), [SSIM](https://en.wikipedia.org/wiki/Structural_similarity) (>0.95 target), [FSIM](https://ieeexplore.ieee.org/document/5705575).
**Statistics:** Two-way [ANOVA](https://en.wikipedia.org/wiki/Analysis_of_variance) (carrier x method) on [AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve) with payload as covariate; [Wilcoxon signed-rank](https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test) for pairwise comparisons; [Cohen's d](https://en.wikipedia.org/wiki/Effect_size#Cohen's_d) effect sizes; [Bonferroni correction](https://en.wikipedia.org/wiki/Bonferroni_correction) (alpha_adj = 0.05/6 ≈ 0.0083) across six hypotheses.

### Experiments

Each research question maps to exactly one experiment; every experiment links back to its RQ.

**Exp. 1 (RQ1: Carrier Origin Effect).** Apply [RS Analysis](https://en.wikipedia.org/wiki/Steganalysis#RS_analysis) and [SRM](https://ieeexplore.ieee.org/document/6197267)+[FLD](https://en.wikipedia.org/wiki/Linear_discriminant_analysis#Fisher's_linear_discriminant) to all 1,000 images embedded at all payload levels and methods. Compute [AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve) per carrier type (conditions A, B). Compare real vs. ML-generated AUC with [Wilcoxon signed-rank test](https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test) and [Cohen's d](https://en.wikipedia.org/wiki/Effect_size#Cohen's_d); apply [Bonferroni correction](https://en.wikipedia.org/wiki/Bonferroni_correction). A significant difference (Bonferroni-corrected p < 0.0083) with |d| > 0.2 confirms H1.

**Exp. 2 (RQ2: Payload Sensitivity).** From Exp. 1 results, plot [AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve) vs. payload level (Low/Medium/High) separately for real and ML-generated carriers, with separate curves per embedding method. Test whether the real-ML AUC gap increases monotonically using [Spearman's rho](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient) on the difference series; test the carrier x payload interaction in [ANOVA](https://en.wikipedia.org/wiki/Analysis_of_variance).

**Exp. 3 (RQ3: Method Interaction).** Run a 2 x 2 two-way [ANOVA](https://en.wikipedia.org/wiki/Analysis_of_variance) with factors carrier origin (real/ML) and embedding method ([LSB](https://en.wikipedia.org/wiki/Least_significant_bit)/[DCT](https://en.wikipedia.org/wiki/Discrete_cosine_transform)) on [SRM](https://ieeexplore.ieee.org/document/6197267) [AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve) scores. A significant interaction ([F-test](https://en.wikipedia.org/wiki/F-test), [Bonferroni-corrected](https://en.wikipedia.org/wiki/Bonferroni_correction)) indicates the method's detectability gap depends on carrier origin, confirming H3.

**Exp. 4 (RQ4: Encryption Effect).** For each carrier type and embedding method, compare [AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve) scores between the plain-payload and [AES-256-CBC](https://en.wikipedia.org/wiki/Advanced_Encryption_Standard)-encrypted-payload conditions. [AES](https://en.wikipedia.org/wiki/Advanced_Encryption_Standard) encryption randomises the bit pattern of the message before embedding; if detector AUC drops significantly ([Wilcoxon signed-rank](https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test), [Bonferroni-corrected](https://en.wikipedia.org/wiki/Bonferroni_correction)), this indicates that message structure contributes to detection beyond purely carrier-level embedding distortion. An interaction with carrier type (real vs. ML) would suggest that generative model statistics moderate the encryption benefit.

### Prototype

**Vertical prototype (algorithm depth):** Isolated implementation and verification of all four core algorithms: (1) [LSB](https://en.wikipedia.org/wiki/Least_significant_bit) embedding and extraction, verified by [BER](https://en.wikipedia.org/wiki/Bit_error_rate) = 0 on a 25-image test set; (2) [DCT](https://en.wikipedia.org/wiki/Discrete_cosine_transform)-[QIM](https://en.wikipedia.org/wiki/Quantization_index_modulation) embedding, verified by lossless payload recovery; (3) [RS Analysis](https://en.wikipedia.org/wiki/Steganalysis#RS_analysis), validated against published estimates on known-cover images; (4) [SRM](https://ieeexplore.ieee.org/document/6197267) feature extraction, verified by reference [AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve) > 0.70 on a 25-cover/25-stego pair set.

**Horizontal prototype (integration breadth):** The four verified algorithms connected into an end-to-end pipeline on a 50-image subset (25 real + 25 ML) at medium [LSB](https://en.wikipedia.org/wiki/Least_significant_bit) payload, validating inter-component interfaces before scaling to the full 1,000-image experiment.

### Related Work

#### Generative Steganography
[Hu et al. (2023)](https://doi.org/10.3390/electronics12051253) and [Liu et al. (2024)](https://doi.org/10.1109/TDSC.2024.3372139) use [GANs](https://en.wikipedia.org/wiki/Generative_adversarial_network) and [diffusion models](https://en.wikipedia.org/wiki/Diffusion_model) *as the embedding mechanism itself*, synthesising images that inherently encode a message without any post-hoc modification. [Duan et al. (2020)](https://doi.org/10.1186/s13640-020-00506-6) developed coverless steganography that generates stego-images from scratch. Our work is fundamentally different: we use ML-generated images purely as *passive carriers* for standard [LSB](https://en.wikipedia.org/wiki/Least_significant_bit)/[DCT](https://en.wikipedia.org/wiki/Discrete_cosine_transform) embedding, without modifying the generation process. The research question of how the carrier's statistical origin affects detectability is orthogonal to generative embedding.

#### AI-Generated Images as Carriers
[De et al. (2022)](https://doi.org/10.1186/s13638-022-02190-8) is the closest prior work, demonstrating steganographic secret sharing via AI-generated photorealistic images using [minimum-entropy coupling](https://en.wikipedia.org/wiki/Coupling_(probability)). However, three key differences distinguish our study: (1) De et al. use a bespoke probabilistic embedding scheme rather than standard [LSB](https://en.wikipedia.org/wiki/Least_significant_bit)/[DCT](https://en.wikipedia.org/wiki/Discrete_cosine_transform); (2) they do not compare real vs. ML-generated carriers side-by-side under controlled conditions; and (3) they do not evaluate [steganalysis](https://en.wikipedia.org/wiki/Steganalysis) detection rates. Our study directly fills these gaps with a [factorial design](https://en.wikipedia.org/wiki/Factorial_experiment) that isolates carrier origin as the independent variable.

#### Cross-Domain Steganalysis
Recent work has studied cross-domain generalisation in [steganalysis](https://en.wikipedia.org/wiki/Steganalysis), specifically training on images from one camera model and testing on another ([Fridrich & Kodovsky, 2012](https://doi.org/10.1109/TIFS.2012.2190402)). [Unsupervised domain adaptation](https://en.wikipedia.org/wiki/Domain_adaptation) and self-supervised approaches have been proposed to bridge this gap. However, none of this work examines the specific shift from natural photographs to ML-generated images, which is qualitatively different from inter-camera variation: it involves a shift in the entire generative process, not merely sensor noise characteristics.

#### Deepfake and Synthetic Image Detection
[Wang et al. (2020)](https://arxiv.org/abs/1912.11035) showed that [CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network)-generated images are surprisingly detectable by simple [linear classifiers](https://en.wikipedia.org/wiki/Linear_classifier), confirming that generative models impose statistical regularities absent from photographs. [Corvi et al. (2023)](https://arxiv.org/abs/2211.00680) extended this to [diffusion-model](https://en.wikipedia.org/wiki/Diffusion_model)-generated images. We leverage these findings (ML-generated images do have different statistical properties from photographs) and apply the same insight to [steganalysis](https://en.wikipedia.org/wiki/Steganalysis) rather than [image forensics](https://en.wikipedia.org/wiki/Digital_image_forensics). The key difference: [deepfake](https://en.wikipedia.org/wiki/Deepfake) detection aims to distinguish real from fake; we aim to understand how this statistical difference affects the *detectability of embedded content*.

#### Classical vs. Deep Steganalysis
State-of-the-art [steganalysis](https://en.wikipedia.org/wiki/Steganalysis) uses [deep residual networks](https://en.wikipedia.org/wiki/Residual_neural_network) achieving near-perfect [AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve) on standard benchmarks ([Luo et al., 2024](https://doi.org/10.1016/j.neucom.2024.127334)). We deliberately choose [SRM](https://ieeexplore.ieee.org/document/6197267)+[FLD](https://en.wikipedia.org/wiki/Linear_discriminant_analysis#Fisher's_linear_discriminant) (classical ML) over these neural approaches for three reasons: (1) it matches our course scope in cryptography/steganography; (2) its hand-crafted features are interpretable and less likely to overfit to carrier-specific artefacts, making it a fairer cross-domain test; (3) it runs on CPU in minutes, making the full 1,000-image study feasible within the project timeline.

### Relation to Curriculum

This project applies core concepts from **Cryptography and Steganography** ([LSB](https://en.wikipedia.org/wiki/Least_significant_bit)/[DCT](https://en.wikipedia.org/wiki/Discrete_cosine_transform) embedding, [AES-256](https://en.wikipedia.org/wiki/Advanced_Encryption_Standard) encryption, information-theoretic detectability), **Research Methods** ([factorial experimental design](https://en.wikipedia.org/wiki/Factorial_experiment), [ANOVA](https://en.wikipedia.org/wiki/Analysis_of_variance), [effect sizes](https://en.wikipedia.org/wiki/Effect_size), [hypothesis testing](https://en.wikipedia.org/wiki/Statistical_hypothesis_testing)), **Machine Learning** ([SRM](https://ieeexplore.ieee.org/document/6197267)+[FLD](https://en.wikipedia.org/wiki/Linear_discriminant_analysis#Fisher's_linear_discriminant) feature-based classification), and **Algorithm Design and Data Structures** ([DCT](https://en.wikipedia.org/wiki/Discrete_cosine_transform) and [QIM](https://en.wikipedia.org/wiki/Quantization_index_modulation) implementation in Python).

### Planning

The project is divided into two phases aligned with the Semester 2 academic calendar. Detailed [Gantt charts](https://en.wikipedia.org/wiki/Gantt_chart) are provided in the midway proposal PDF (`docs/proposals/midway_proposal.pdf`).

**Phase 2: Implementation** (Period 5, 30 Mar - 15 May 2026, 7 weeks) covers three parallel workstreams: dataset construction and ML image generation (Wk 1-2), steganography pipeline implementation including [LSB](https://en.wikipedia.org/wiki/Least_significant_bit), [DCT](https://en.wikipedia.org/wiki/Discrete_cosine_transform), and [AES-256](https://en.wikipedia.org/wiki/Advanced_Encryption_Standard) encryption (Wk 2-3), and detection, analysis, and writing (Wk 3-7).

**Phase 3: Completion** (Project Period, 25 May - 12 Jun 2026, 3 weeks) covers completing any remaining implementation, verifying and rerunning experiments, and finalising all deliverables (presentation slides, poster, and paper).

### Minimal Passing Requirements

**Product:** Functional [LSB](https://en.wikipedia.org/wiki/Least_significant_bit) and [DCT](https://en.wikipedia.org/wiki/Discrete_cosine_transform) embedding pipelines (plain and [AES-256](https://en.wikipedia.org/wiki/Advanced_Encryption_Standard) encrypted); [RS Analysis](https://en.wikipedia.org/wiki/Steganalysis#RS_analysis) and [SRM](https://ieeexplore.ieee.org/document/6197267)+[FLD](https://en.wikipedia.org/wiki/Linear_discriminant_analysis#Fisher's_linear_discriminant) detectors evaluated on all 1,000 images across both encryption conditions.

**Validation:** RQ1 and RQ4 answered with [significance tests](https://en.wikipedia.org/wiki/Statistical_significance) on [AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve) across carrier types and encryption conditions; null results characterised with 95% [CIs](https://en.wikipedia.org/wiki/Confidence_interval).

### References

1. Petitcolas, F. A. P., Anderson, R. J., & Kuhn, M. G. (1999). [Information hiding: A survey.](https://doi.org/10.1109/5.771065) *Proceedings of the IEEE*, 87(7), 1062-1078.
2. Cheddad, A., Condell, J., Curran, K., & McKevitt, P. (2010). [Digital image steganography: Survey and analysis of current methods.](https://doi.org/10.1016/j.sigpro.2009.08.010) *Signal Processing*, 90(3), 727-752.
3. Hussain, M., Wahab, A. W. A., Idris, Y. I. B., Ho, A. T. S., & Jung, K. H. (2018). [Image steganography in spatial domain: A survey.](https://doi.org/10.1016/j.image.2018.03.012) *Signal Processing: Image Communication*, 65, 46-66.
4. Fridrich, J., & Kodovsky, J. (2012). [Rich models for steganalysis of digital images.](https://doi.org/10.1109/TIFS.2012.2190402) *IEEE Transactions on Information Forensics and Security*, 7(3), 868-882.
5. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). [High-resolution image synthesis with latent diffusion models.](https://arxiv.org/abs/2112.10752) In *Proceedings of the IEEE/CVF CVPR* (pp. 10684-10695).
6. Karras, T., Laine, S., Aittala, M., Hellsten, J., Lehtinen, J., & Aila, T. (2021). [Alias-free generative adversarial networks.](https://proceedings.neurips.cc/paper/2021/hash/076ccd93ad68be51f23707988e934906-Abstract.html) In *Proceedings of NeurIPS*, 34, 852-863.
7. De, A., Kinzel, W., & Kanter, I. (2022). [Steganographic secret sharing via AI-generated photorealistic images.](https://doi.org/10.1186/s13638-022-02190-8) *EURASIP Journal on Wireless Communications and Networking*, art. 108.
8. Fridrich, J., Goljan, M., & Du, R. (2001). [Detecting LSB steganography in color and grayscale images.](https://doi.org/10.1109/93.959097) *IEEE Multimedia*, 8(4), 22-28.
9. Westfeld, A., & Pfitzmann, A. (1999). [Attacks on steganographic systems.](https://doi.org/10.1007/10719724_5) In *Proceedings of the 3rd International Workshop on Information Hiding*, LNCS 1768 (pp. 61-76).
10. Chen, B., & Wornell, G. W. (2001). [Quantization index modulation: A class of provably good methods for digital watermarking and information embedding.](https://doi.org/10.1109/18.923725) *IEEE Transactions on Information Theory*, 47(4), 1423-1443.
11. Wang, S. Y., Wang, O., Zhang, R., Owens, A., & Efros, A. A. (2020). [CNN-generated images are surprisingly easy to spot... for now.](https://arxiv.org/abs/1912.11035) In *Proceedings of the IEEE/CVF CVPR* (pp. 8695-8704).
12. Corvi, R., Cozzolino, D., Zingarini, G., Poggi, G., Nagano, K., & Verdoliva, L. (2023). [On the detection of synthetic images generated by diffusion models.](https://arxiv.org/abs/2211.00680) In *Proceedings of IEEE ICASSP* (pp. 1-5).
13. Dang-Nguyen, D.-T., Pasquini, C., Conotter, V., & Boato, G. (2015). [RAISE: A raw images dataset for digital image forensics.](https://doi.org/10.1145/2713168.2713194) In *Proceedings of ACM MMSys* (pp. 219-224).
14. Lin, T.-Y., et al. (2014). [Microsoft COCO: Common objects in context.](https://arxiv.org/abs/1405.0312) In *Proceedings of ECCV*, LNCS 8693 (pp. 740-755).
15. Young, P., Lai, A., Hodosh, M., & Hockenmaier, J. (2014). [From image descriptions to visual denotations.](https://doi.org/10.1162/tacl_a_00166) *Transactions of the Association for Computational Linguistics*, 2, 67-78.
16. Luo, Y., et al. (2024). [Deep learning for steganalysis of diverse data types: A review.](https://doi.org/10.1016/j.neucom.2024.127334) *Neurocomputing*, Elsevier.
17. Hu, P., et al. (2023). [A coverless image steganography based on generative adversarial networks.](https://doi.org/10.3390/electronics12051253) *Electronics*, 12(5), art. 1253.
18. Liu, X., et al. (2024). [Message-driven generative steganography using GAN.](https://doi.org/10.1109/TDSC.2024.3372139) *IEEE Transactions on Dependable and Secure Computing*.
19. Duan, X., Jia, D., Li, B., Guo, D., Zhang, E., & Qin, C. (2020). [Coverless steganography based on generative adversarial network.](https://doi.org/10.1186/s13640-020-00506-6) *EURASIP Journal on Image and Video Processing*, art. 46.

---

## Implementation

Detailed implementation specifications, code examples, and experimental protocols.

### Table of Contents

1. [Overview](#1-overview)
2. [Dataset Construction (Phase 1)](#2-dataset-construction-phase-1)
3. [Steganographic Embedding (Phase 2)](#3-steganographic-embedding-phase-2)
4. [Steganalysis Detectors (Phase 3)](#4-steganalysis-detectors-phase-3)
5. [Experimental Conditions (Phase 4)](#5-experimental-conditions-phase-4)
6. [Evaluation Metrics (Phase 5)](#6-evaluation-metrics-phase-5)
7. [Expected Results Structure](#7-expected-results-structure)
8. [Division of Labor & Timeline](#8-division-of-labor--timeline)
9. [References](#9-references)

---

### 1. Overview

This project investigates whether the **origin of a carrier image** (real photographs captured by humans versus images synthesized by generative ML models) meaningfully affects the detectability of hidden payloads embedded via image steganography. All embedding procedures, payload sizes, and detection classifiers are held constant; the sole independent variable of primary interest is carrier origin.

The study is motivated by the rapid proliferation of AI-generated imagery. Steganalysis tools are trained and benchmarked almost exclusively on photographic datasets. If ML-generated images possess statistically distinct noise statistics, texture regularity, or spectral characteristics compared to photographs, then classical and learned steganalysis models may behave asymmetrically, either over-detecting or under-detecting hidden content depending on carrier source. This has direct implications for security, watermarking, and forensic practice.

#### Factorial Design

The full factorial design has the following structure:

**2 x 2 x 3 x 2**

| Factor | Levels | Values |
|---|---|---|
| Carrier origin | 2 | Real photographs, ML-generated |
| Embedding method | 2 | LSB substitution, DCT-based |
| Payload size | 3 | Low (~0.08 bpp), Medium (~0.16 bpp), High (~0.32 bpp) |
| Payload encryption | 2 | Plain, AES-256-CBC encrypted |

Detectors: RS Analysis + χ² (training-free), SRM + FLD ensemble (classical ML). Cross-domain conditions (A-E) apply to SRM only.

---

### 2. Dataset Construction (Phase 1)

#### 2.1 Real Images

**Sources:**
- **RAISE** (Dang-Nguyen et al., 2015): RAW, high-resolution, unprocessed photos from a variety of cameras. Download from `http://loki.disi.unitn.it/RAISE/`. Select 250 images stratified by scene category (outdoor, indoor, portrait, macro). RAISE is preferred as the primary source because its RAW format preserves camera sensor noise structure, the natural image statistics that steganalysis exploits.
- **COCO** (Lin et al., 2014): Use COCO validation split for 150 images, selected to match the scene categories above. Provides caption annotations that feed directly into the matching protocol (Section 2.3).
- **Flickr30k** (Young et al., 2014): Use Flickr30k test split for 100 images. Also provides caption annotations for SD prompt generation.

**Normalization target:** 512×512 px, RGB, 8-bit depth, lossless PNG. RAISE images must be demosaiced from RAW before normalization.

```python
import rawpy
import imageio
from PIL import Image
import numpy as np
from pathlib import Path


def normalize_real_image(input_path: str, output_path: str, size: int = 512) -> None:
    """
    Load a real camera image (RAW or standard format), normalize to
    512x512 RGB 8-bit PNG. Handles both RAW (RAISE) and JPEG/PNG (COCO/Flickr).
    """
    input_path = Path(input_path)
    suffix = input_path.suffix.lower()

    raw_extensions = {".nef", ".cr2", ".arw", ".dng", ".rw2"}

    if suffix in raw_extensions:
        with rawpy.imread(str(input_path)) as raw:
            # postprocess returns uint8 or uint16; specify 8-bit output
            rgb = raw.postprocess(
                use_camera_wb=True,
                output_bps=8,
                no_auto_bright=False,
            )
        img = Image.fromarray(rgb, mode="RGB")
    else:
        img = Image.open(input_path).convert("RGB")

    # Center-crop to square, then resize
    w, h = img.size
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    img = img.crop((left, top, left + min_dim, top + min_dim))
    img = img.resize((size, size), Image.LANCZOS)

    # Confirm 8-bit
    arr = np.array(img)
    assert arr.dtype == np.uint8, "Output must be uint8"
    assert arr.shape == (size, size, 3), f"Shape mismatch: {arr.shape}"

    img.save(output_path, format="PNG", optimize=False)
```

#### 2.2 ML-Generated Images

**Stable Diffusion v2.1** (250 images) using the HuggingFace `diffusers` library (via Apple MPS or CUDA). Prompts are derived directly from COCO/Flickr30k captions associated with the real images selected above, ensuring semantic alignment (see Section 2.3).

```python
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from pathlib import Path


def generate_sd_images(
    prompts: list[str],
    output_dir: str,
    model_id: str = "stabilityai/stable-diffusion-2-1",
    seed: int = 42,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
) -> None:
    """
    Generate images using Stable Diffusion v2.1.
    Saves normalized 512x512 RGB PNG files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()

    generator = torch.Generator(device="cuda").manual_seed(seed)

    for idx, prompt in enumerate(prompts):
        result = pipe(
            prompt,
            height=512,
            width=512,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        img = result.images[0]  # PIL Image, already 512x512
        fname = output_dir / f"sd21_{idx:04d}.png"
        img.save(fname, format="PNG")
        print(f"[{idx+1}/{len(prompts)}] Saved {fname}")
```

**StyleGAN3** (250 images): Use the official NVIDIA implementation (`https://github.com/NVlabs/stylegan3`). Generate with `--seeds` parameter spanning a fixed range for reproducibility. Convert output `.png` files to the normalized format using the same `normalize_real_image` function above (already 512×512 output by default).

```bash
# Example StyleGAN3 generation command (run in the stylegan3/ repo directory)
python gen_images.py \
  --outdir=out/stylegan3 \
  --trunc=0.7 \
  --seeds=0-249 \
  --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhqu-256x256.pkl
```

#### 2.3 Matching Protocol

Semantic consistency between real and ML-generated images is critical for a fair comparison. The matching procedure is:

1. **Caption-to-prompt mapping:** For each real COCO/Flickr30k image, use its associated human caption directly as the Stable Diffusion prompt. For RAISE images (no captions), generate a short descriptive caption using BLIP-2 before passing it to SD.
2. **Format normalization:** All images (real and generated) must pass through `normalize_real_image` to guarantee identical dimensions, color space, and bit depth.
3. **Quality gate:** Compute BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator) for every image. Reject any image with BRISQUE > 50 and regenerate or resample.

```python
import piq
import torch
from PIL import Image
import numpy as np


def passes_quality_gate(image_path: str, brisque_threshold: float = 50.0) -> bool:
    """
    Return True if the image passes the BRISQUE quality gate.
    Lower BRISQUE = better perceptual quality.
    """
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    # piq expects (B, C, H, W) tensor in [0, 1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    score = piq.brisque(tensor, data_range=1.0).item()
    return score < brisque_threshold
```

4. **File naming convention:** All images in the final dataset must follow:

```
{source}_{id:04d}_{category}.png
```

Where `source` ∈ `{raise, coco, flickr, sd21, sg3}`, `id` is a zero-padded 4-digit integer, and `category` ∈ `{outdoor, indoor, portrait, macro, other}`.

Examples: `raise_0001_outdoor.png`, `sd21_0042_portrait.png`.

#### 2.4 Dataset Summary Table

| Source | n_images | Format | Notes |
|---|---|---|---|
| RAISE | 250 | 512×512 RGB PNG | Demosaiced from RAW; unprocessed originals |
| COCO val2017 | 150 | 512×512 RGB PNG | Center-cropped; captions used as SD prompts |
| Flickr30k test | 100 | 512×512 RGB PNG | Center-cropped; captions used as SD prompts |
| Stable Diffusion v2.1 | 250 | 512×512 RGB PNG | Prompts from COCO/Flickr captions; seed=42 |
| StyleGAN3 | 250 | 512×512 RGB PNG | Seeds 0–249; truncation ψ=0.7 |
| **Total** | **1,000** | | 500 real + 500 ML-generated |

---

### 3. Steganographic Embedding (Phase 2)

#### 3.1 LSB Substitution

**Algorithm:** The $k$-LSB substitution method replaces the $k$ least significant bits of a selected pixel channel value with $k$ bits of the message. For a pixel value $x \in [0, 255]$ and a $k$-bit message chunk $m$:

$$x' = (x \;\&\; \sim\text{mask}_k) \;|\; m$$

where $\text{mask}_k = 2^k - 1$ and $\sim$ denotes bitwise NOT. The operation zeroes out the $k$ LSBs of $x$ and inserts $m$ in their place. Perceptual distortion scales as $O(2^k)$; for $k=1$ the maximum channel change is 1 LSU (least significant unit), imperceptible to the human visual system.

Pixel selection is pseudorandom, governed by a PRNG seeded with a secret key $K$, so that only the key-holder can locate and extract the embedded bits. The PRNG produces a permutation of all valid pixel-channel indices; embedding proceeds along this permuted order until the payload is exhausted.

```python
import numpy as np
from PIL import Image
from typing import Optional


def lsb_embed(
    cover_path: str,
    payload_bits: np.ndarray,
    output_path: str,
    k: int = 1,
    pixel_fraction: float = 0.5,
    key: int = 12345,
) -> None:
    """
    Embed payload_bits into a cover image using k-LSB substitution.

    Args:
        cover_path:     Path to the cover image (PNG, RGB, 512x512).
        payload_bits:   1-D numpy array of bits (dtype uint8, values 0 or 1).
        output_path:    Path to save the stego image.
        k:              Number of LSBs to replace per pixel-channel (1, 2, or 4).
        pixel_fraction: Fraction of all pixel-channel slots to use (0 < f <= 1).
        key:            Integer seed for the PRNG controlling pixel selection order.
    """
    img = Image.open(cover_path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)  # (H, W, 3)
    H, W, C = arr.shape

    # Generate pseudorandom pixel-channel index permutation
    rng = np.random.default_rng(seed=key)
    total_slots = H * W * C
    n_slots = int(total_slots * pixel_fraction)
    selected_indices = rng.choice(total_slots, size=n_slots, replace=False)

    # Pack payload_bits into k-bit chunks
    # Pad to multiple of k
    pad_len = (-len(payload_bits)) % k
    padded = np.concatenate([payload_bits, np.zeros(pad_len, dtype=np.uint8)])
    chunks = padded.reshape(-1, k)
    # Convert each k-bit row to integer value
    powers = (2 ** np.arange(k - 1, -1, -1)).astype(np.uint8)
    chunk_values = (chunks * powers).sum(axis=1).astype(np.uint8)

    if len(chunk_values) > n_slots:
        raise ValueError(
            f"Payload too large: need {len(chunk_values)} slots, have {n_slots}"
        )

    mask_k = np.uint8(2**k - 1)
    flat = arr.flatten()

    for slot_i, chunk_val in zip(selected_indices, chunk_values):
        flat[slot_i] = np.uint8((int(flat[slot_i]) & ~int(mask_k)) | int(chunk_val))

    stego_arr = flat.reshape(H, W, C)
    stego_img = Image.fromarray(stego_arr, mode="RGB")
    stego_img.save(output_path, format="PNG")
```

#### 3.2 DCT-Based Embedding

**Algorithm:** The image is partitioned into non-overlapping 8×8 pixel blocks. For each block, a 2D Discrete Cosine Transform (DCT-II) is applied, yielding 64 frequency coefficients. Mid-frequency coefficients at zigzag scan positions 10–54 are selected for embedding; these are perceptually less critical than DC and low-frequency components but more robust than high-frequency components that are highly sensitive to noise.

Embedding uses **Quantization Index Modulation (QIM)**. For a coefficient $C_i$, step size $\Delta$, and message bit $b \in \{0, 1\}$:

$$C'_i = \Delta \cdot \operatorname{round}\!\left(\frac{C_i}{\Delta}\right) + \begin{cases} +\Delta/4 & \text{if } b = 1 \\ -\Delta/4 & \text{if } b = 0 \end{cases}$$

This places $C'_i$ in the center of the appropriate quantization bin, providing robustness to small perturbations (Chen & Wornell, 2001).

```python
import numpy as np
from PIL import Image
from scipy.fft import dctn, idctn


# Zigzag scan order for 8x8 block (precomputed flat indices)
def _zigzag_indices(n: int = 8) -> list[int]:
    result = []
    for s in range(2 * n - 1):
        if s % 2 == 0:
            r_range = range(min(s, n - 1), max(-1, s - n), -1)
        else:
            r_range = range(max(0, s - n + 1), min(s + 1, n))
        for r in r_range:
            c = s - r
            if 0 <= c < n:
                result.append(r * n + c)
    return result


ZIGZAG = _zigzag_indices(8)
MID_FREQ_POSITIONS = ZIGZAG[10:55]  # 45 positions


def dct_embed(
    cover_path: str,
    payload_bits: np.ndarray,
    output_path: str,
    coeff_fraction: float = 0.25,
    delta: float = 20.0,
    channel: int = 0,
) -> None:
    """
    Embed payload_bits into a cover image using DCT + QIM.

    Args:
        cover_path:      Path to the cover image (PNG, RGB, 512x512).
        payload_bits:    1-D numpy array of bits (dtype uint8, values 0 or 1).
        output_path:     Path to save the stego image.
        coeff_fraction:  Fraction of mid-frequency coefficients to use per block.
        delta:           QIM step size. Larger delta = higher robustness, more distortion.
        channel:         Which RGB channel to use for embedding (0=R, 1=G, 2=B).
    """
    img = Image.open(cover_path).convert("RGB")
    arr = np.array(img, dtype=np.float64)
    H, W, C = arr.shape
    assert H % 8 == 0 and W % 8 == 0, "Image dimensions must be multiples of 8"

    n_mid = len(MID_FREQ_POSITIONS)
    n_use = max(1, int(n_mid * coeff_fraction))
    use_positions = MID_FREQ_POSITIONS[:n_use]

    stego_arr = arr.copy()
    ch = arr[:, :, channel]

    bit_idx = 0
    n_bits = len(payload_bits)

    for row in range(0, H, 8):
        for col in range(0, W, 8):
            if bit_idx >= n_bits:
                break
            block = ch[row:row+8, col:col+8]
            coeffs = dctn(block, norm="ortho")
            flat_coeffs = coeffs.flatten()

            for pos in use_positions:
                if bit_idx >= n_bits:
                    break
                b = int(payload_bits[bit_idx])
                ci = flat_coeffs[pos]
                quantized = delta * round(ci / delta)
                flat_coeffs[pos] = quantized + (delta / 4 if b == 1 else -delta / 4)
                bit_idx += 1

            coeffs = flat_coeffs.reshape(8, 8)
            block_stego = idctn(coeffs, norm="ortho")
            stego_arr[row:row+8, col:col+8, channel] = block_stego

    stego_arr = np.clip(stego_arr, 0, 255).astype(np.uint8)
    stego_img = Image.fromarray(stego_arr, mode="RGB")
    stego_img.save(output_path, format="PNG")
```

#### 3.3 AES-256 Encryption (RQ4)

Before embedding, the payload can optionally be encrypted with AES-256-CBC. This is a critical experimental variable because AES output is statistically indistinguishable from a uniform random bitstream: it has maximum entropy and no exploitable structure. The **H6 hypothesis** therefore predicts that encryption has no significant effect on detectability: if steganalysis tools detect steganography by sensing the presence of a non-natural signal (regardless of whether it has linguistic or structured content), then encrypting a random-looking payload should not change detectability. Conversely, if certain classifiers are sensitive to the statistical structure of natural-language payloads, H6 would be disconfirmed.

```python
import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import numpy as np


def encrypt_payload(plaintext_bits: np.ndarray, key: bytes) -> np.ndarray:
    """
    Encrypt a bitstream with AES-256-CBC. Returns encrypted bits.

    Args:
        plaintext_bits: 1-D uint8 numpy array of bits (0 or 1).
        key:            32-byte AES-256 key.

    Returns:
        Encrypted bitstream as 1-D uint8 numpy array of bits.
    """
    assert len(key) == 32, "AES-256 requires a 32-byte key"

    # Pack bits into bytes
    pad_len = (-len(plaintext_bits)) % 8
    padded = np.concatenate([plaintext_bits, np.zeros(pad_len, dtype=np.uint8)])
    plaintext_bytes = np.packbits(padded).tobytes()

    iv = os.urandom(16)
    cipher = Cipher(
        algorithms.AES(key),
        modes.CBC(iv),
        backend=default_backend(),
    )
    encryptor = cipher.encryptor()

    # PKCS7 padding to block size
    block_size = 16
    pad_bytes = block_size - (len(plaintext_bytes) % block_size)
    padded_bytes = plaintext_bytes + bytes([pad_bytes] * pad_bytes)

    ciphertext = encryptor.update(padded_bytes) + encryptor.finalize()
    # Prepend IV for decryption
    full_output = iv + ciphertext

    encrypted_bits = np.unpackbits(np.frombuffer(full_output, dtype=np.uint8))
    return encrypted_bits.astype(np.uint8)
```

The payload used in experiments is a pseudorandom bitstream (generated via a fixed-seed PRNG) to eliminate any content-dependence. When encryption is enabled, this bitstream is passed through `encrypt_payload` first.

#### 3.4 Payload Rates

| Level | LSB config | DCT config | Approx bpp |
|---|---|---|---|
| Low | k=1, 25% pixels | 10% coefficients | ~0.08 bpp |
| Medium | k=1, 50% pixels | 25% coefficients | ~0.16 bpp |
| High | k=2, 50% pixels | 50% coefficients | ~0.32 bpp |

Bits per pixel (bpp) is computed as: $\text{bpp} = k \times f_{\text{pixels}}$ for LSB, and $\text{bpp} = \frac{n_{\text{mid}} \times f_{\text{coeffs}}}{64}$ for DCT (one bit per selected coefficient per block of 64 pixels).

#### 3.5 Full Embedding Matrix

**Cover images:** 500 real + 500 ML-generated = **1,000 covers**

**Multiplication:**

$$1{,}000 \times 2_{\text{methods}} \times 3_{\text{payload}} \times 2_{\text{encryption}} = \mathbf{12{,}000 \text{ stego images}}$$

Plus 1,000 unmodified covers = **13,000 total images on disk**.

**Directory structure** (already created in `data/`):

```
data/
├── covers/
│   ├── real/
│   │   ├── raise_0001_outdoor.png
│   │   └── ...
│   └── ml/
│       ├── sd21_0001_outdoor.png
│       └── ...
└── stego/
    ├── lsb/
    │   ├── low/
    │   │   ├── plain/
    │   │   │   ├── real/
    │   │   │   └── ml/
    │   │   └── encrypted/
    │   │       ├── real/
    │   │       └── ml/
    │   ├── medium/
    │   │   └── ... (same plain/encrypted × real/ml structure)
    │   └── high/
    │       └── ...
    └── dct/
        ├── low/
        ├── medium/
        └── high/
            └── ... (same structure as lsb/)
```

Each leaf directory contains stego `.png` files named identically to their cover counterparts, enabling straightforward pair-matching during evaluation.

---

### 4. Steganalysis Detectors (Phase 3)

The study uses two detectors deliberately chosen to stay within the scope of classical signal processing and statistics, avoiding deep learning entirely.

| Detector | Type | Training required? | Estimated compute |
|---|---|---|---|
| RS Analysis | Training-free statistical test | None | Seconds per image |
| SRM + FLD ensemble | Classical ML (no neural networks) | Yes (CPU only, fast) | Minutes total |

---

#### 4.1 RS Analysis (Training-Free)

RS Analysis (Fridrich, Goljan & Du, 2001) is a **fully analytical method** requiring no training data, no labels, and no classifier. It detects LSB embedding by exploiting the fact that steganographic embedding disturbs the natural regularity structure of pixel neighborhoods.

**How it works:** The image is divided into pixel groups (e.g., sets of 4 adjacent pixels). Each group is classified as *Regular* (R), *Singular* (S), or *Unusable* (U) according to a noise-sensitivity function $f$. The key insight is that for a clean image, $R_M \approx R_{-M}$ and $S_M \approx S_{-M}$ (where $M$ and $-M$ are original and inverted flip masks). LSB embedding breaks this symmetry. By solving the resulting system of equations, the **embedding rate** $p$ can be estimated directly:

$$p \approx \frac{2(R_{-M} - R_M)}{(R_{-M} - R_M) + (S_{-M} - S_M)}$$

This gives not just a binary detect/not-detect decision, but an **estimate of the payload fraction** embedded, useful for RQ1 (payload sensitivity).

```python
import numpy as np
from PIL import Image


def rs_analysis(image_path: str, mask: list[int] = [0, 1, 1, 0]) -> dict:
    """
    RS Analysis for LSB steganography detection.
    Reference: Fridrich, Goljan & Du (2001), SPIE Proc.

    Args:
        image_path: Path to the image under analysis (PNG, RGB or grayscale).
        mask:       Flip mask M (length must divide evenly into image rows).

    Returns:
        Dictionary with R_M, S_M, R_-M, S_-M, and estimated payload fraction p_hat.
    """
    img = Image.open(image_path).convert("L")
    arr = np.array(img, dtype=np.int32)
    H, W = arr.shape
    m = len(mask)
    mask = np.array(mask, dtype=np.int32)

    def flip(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply flip operation F_1: even <-> odd for mask positions."""
        result = x.copy()
        for i, bit in enumerate(mask):
            if bit == 1:
                # XOR LSB: 0->1, 1->0 (flip between even/odd)
                result[i] = x[i] ^ 1
        return result

    def neg_flip(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply negative flip F_{-1}: shift by -1 with wrap."""
        result = x.copy()
        for i, bit in enumerate(mask):
            if bit == 1:
                result[i] = (x[i] - 1) % 256 if x[i] % 2 == 0 else (x[i] + 1) % 256
        return result

    def noise_func(group: np.ndarray) -> int:
        """Measure noise (sum of absolute differences of adjacent pixels)."""
        return int(np.sum(np.abs(np.diff(group))))

    R_M = S_M = R_nM = S_nM = 0
    total_groups = 0

    for row in range(H):
        for col in range(0, W - m + 1, m):
            group = arr[row, col:col + m]
            if len(group) < m:
                continue
            total_groups += 1

            f_orig = noise_func(group)
            f_flip = noise_func(flip(group, mask))
            f_nflip = noise_func(neg_flip(group, mask))

            if f_flip > f_orig:
                R_M += 1
            elif f_flip < f_orig:
                S_M += 1

            if f_nflip > f_orig:
                R_nM += 1
            elif f_nflip < f_orig:
                S_nM += 1

    R_M /= total_groups
    S_M /= total_groups
    R_nM /= total_groups
    S_nM /= total_groups

    denom = (R_nM - R_M) + (S_nM - S_M)
    p_hat = 2 * (R_nM - R_M) / denom if abs(denom) > 1e-9 else 0.0

    return {
        "R_M": R_M, "S_M": S_M,
        "R_negM": R_nM, "S_negM": S_nM,
        "p_hat": float(np.clip(p_hat, 0, 1)),
        "detected": p_hat > 0.05,  # threshold: >5% payload = detected
    }
```

**Interpretation:** `p_hat > 0` on a cover image = false positive (expected ~0). `p_hat` close to the true embedding rate = correct detection. The detection threshold (default 5%) can be varied to compute ROC curves.

---

#### 4.2 Chi-Square Attack (Training-Free)

The chi-square attack (Westfeld & Pfitzmann, 1999) detects sequential LSB embedding by testing whether pixel value pairs `(2k, 2k+1)` have equal frequency, a property that LSB substitution enforces but natural images do not exhibit.

For a clean image, pixel values 2k and 2k+1 will have different frequencies $n_{2k}$ and $n_{2k+1}$. After LSB embedding into a fraction $p$ of pixels, their frequencies converge toward a common value $\nu_i = (n_{2k} + n_{2k+1}) / 2$. The test statistic is:

$$\chi^2 = \sum_{i=0}^{127} \frac{(\nu_i - n_{2i+1})^2}{\nu_i}$$

Under the null hypothesis (no embedding), this follows a $\chi^2$ distribution with 127 degrees of freedom. A p-value near 1 indicates embedding; near 0 indicates a clean image.

```python
import numpy as np
from PIL import Image
from scipy.stats import chi2


def chi_square_attack(image_path: str, channel: int = 0) -> dict:
    """
    Chi-square attack for LSB steganography detection.
    Reference: Westfeld & Pfitzmann (1999), Information Hiding Workshop.

    Args:
        image_path: Path to image (PNG, RGB).
        channel:    RGB channel index to analyse (0=R, 1=G, 2=B).

    Returns:
        Dictionary with chi2 statistic, p_value, and detection flag.
    """
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)[:, :, channel].flatten().astype(np.int32)

    # Count frequencies of each pixel value 0–255
    counts = np.bincount(arr, minlength=256)

    # Pair up (2k, 2k+1) → expected value ν_i = (n_{2k} + n_{2k+1}) / 2
    evens = counts[0::2]   # n_{2k}
    odds  = counts[1::2]   # n_{2k+1}
    nu    = (evens + odds) / 2.0

    # Chi-square statistic (avoid division by zero)
    valid = nu > 0
    chi2_stat = np.sum((nu[valid] - odds[valid]) ** 2 / nu[valid])
    df = int(valid.sum()) - 1

    p_value = chi2.sf(chi2_stat, df=df)  # survival function = 1 - CDF

    return {
        "chi2_stat": float(chi2_stat),
        "df": df,
        "p_value": float(p_value),
        "detected": p_value > 0.95,  # high p-value = likely embedded
    }
```

**Note:** The chi-square attack is most effective for sequential LSB embedding. Our pseudorandom pixel selection (keyed embedding) partially defeats it, which is itself an interesting finding to report.

---

#### 4.3 SRM (Spatial Rich Model, Classical ML)

SRM (Fridrich & Kodovský, 2012) sits between training-free methods and deep learning. It applies a bank of high-pass filters to extract residuals, computes co-occurrence statistics (yielding a ~35,000-dimensional feature vector), and classifies with an **ensemble of Fisher Linear Discriminants**, a classical statistical classifier, not a neural network. It requires labeled training data but runs entirely on CPU in minutes.

SRM is included because it handles **DCT-domain embedding** better than the training-free methods above (which are tuned for spatial LSB). It also provides a cross-domain generalization data point: because its features are hand-crafted rather than learned, it may generalize across the real/ML boundary better than a neural network would (addressing RQ3/H4).

```python
import numpy as np
from PIL import Image
import scipy.ndimage
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from typing import Tuple


def extract_srm_features(image_path: str) -> np.ndarray:
    """
    Extract SRM-style residual co-occurrence features.
    Uses a subset of SRM kernels; for the full 35,000-dim version use
    the reference implementation at http://dde.binghamton.edu/download/feature_extractors/
    """
    img = Image.open(image_path).convert("L")
    arr = np.array(img, dtype=np.float32)

    kernels = [
        np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype=np.float32),
        np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]], dtype=np.float32),
        np.array([[0, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32),
        np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32) / 4,
    ]

    features = []
    T = 2
    for kernel in kernels:
        residual = scipy.ndimage.convolve(arr, kernel)
        residual = np.clip(np.round(residual), -T, T).astype(np.int32)
        hist, _ = np.histogram(residual, bins=range(-T, T + 2))
        features.append(hist.astype(np.float32))

    return np.concatenate(features)


def train_srm_classifier(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 3,
) -> Tuple[Pipeline, np.ndarray]:
    """
    Train SRM + FLD-ensemble (approximated as SGDClassifier).
    Returns (fitted pipeline, per-fold AUC scores).
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SGDClassifier(loss="log_loss", max_iter=300, random_state=0)),
    ])
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    pipeline.fit(X, y)
    return pipeline, scores
```

---

#### 4.4 Evaluation Protocol

**Training-free detectors (RS Analysis, Chi-square):**
- Applied directly to every image in the dataset, no training loop, no splits
- Output a continuous score (`p_hat` or `p_value`) per image
- ROC curve computed by sweeping detection threshold across all images
- Cross-domain behavior is automatic: the same analytical formula applies regardless of carrier origin; any difference in detection rate between real and ML images is purely a function of the carrier's statistical properties

**SRM:**
- Requires labeled cover/stego pairs for training
- 3-fold stratified cross-validation (stratified by `{source}_{category}`)
- For cross-domain conditions C and D: train on all of one domain, test on all of the other (no CV needed, see Section 5.3)
- Hardware: CPU only, parallelizable with `n_jobs=-1`

---

### 5. Experimental Conditions (Phase 4)

#### 5.1 Train/Test Conditions

| Condition | Train on | Test on | Addresses | Purpose |
|---|---|---|---|---|
| A | Real | Real | Baseline | Standard detection performance on photographs |
| B | ML-gen | ML-gen | Primary RQ | Within-domain performance on synthetic images |
| C | Real | ML-gen | RQ3, H4/H5 | Cross-domain: real-trained detector on ML images |
| D | ML-gen | Real | RQ3, H4/H5 | Cross-domain: ML-trained detector on real images |
| E | Mixed 50/50 | Both | Mitigation | Does mixed training close the domain gap? |

> **Note:** For the training-free detectors (RS Analysis, chi-square), conditions A–E collapse to a single evaluation: the same formula is applied to all images regardless of carrier origin. The "condition" distinction only applies to **SRM** (which requires a training set). This is itself a notable result: training-free methods are inherently domain-agnostic.

#### 5.2 Full Experiment Matrix

$$2_{\text{detectors}} \times 2_{\text{methods}} \times 3_{\text{payload}} \times 5_{\text{conditions}} = \mathbf{60 \text{ unique configurations}}$$

**SRM training runs** (3-fold CV, only for conditions A, B, E; C and D need no CV):

$$1_{\text{SRM}} \times 2_{\text{methods}} \times 3_{\text{payload}} \times 3_{\text{conditions (A,B,E)}} \times 3_{\text{folds}} = \mathbf{54 \text{ SRM training runs}}$$

**Training-free detectors** (RS Analysis, chi-square): **0 training runs**, applied analytically to all 1,000 images.

**Compute estimates:**

| Detector | Per image | All 1,000 images | All configs |
|---|---|---|---|
| RS Analysis | ~2 sec | ~33 min | ~33 min total |
| Chi-square | < 1 sec | ~8 min | ~8 min total |
| SRM (54 training runs) | ~5 min/fold | ~270 min total | **~4.5 hrs (CPU)** |
| **Grand total** | | | **< 6 hrs** |

#### 5.3 Structuring Test Sets

**Training-free detectors (RS Analysis, chi-square):** No train/test split needed. Apply to every image and record the score (`p_hat` or `p_value`). Conditions A–E are evaluated by simply filtering results to the relevant image subsets after the fact.

**SRM, within-domain conditions (A, B):** 3-fold stratified cross-validation on same-domain data (stratified by `{source}_{scene_category}`). Folds are precomputed and reused across all payload levels and embedding methods.

**SRM, cross-domain conditions (C, D):** Train on **all** images from one domain; test on **all** images from the other. No CV needed; the two domains are fully disjoint by definition. Report a single AUC per configuration.

**SRM, mixed condition (E):** Training set is a 50/50 balanced blend (250 real + 250 ML). Test separately on the held-out real and ML subsets from conditions A and B respectively.

---

### 6. Evaluation Metrics (Phase 5)

#### 6.1 Detection Metrics

- **ROC-AUC (primary):** Area under the receiver operating characteristic curve. Threshold-independent; ranges from 0.5 (random) to 1.0 (perfect). Primary metric for all statistical comparisons.
- **Accuracy at optimal threshold:** Threshold selected by **Youden's J statistic** $J = \text{Sensitivity} + \text{Specificity} - 1$, maximized over all thresholds.
- **EER (Equal Error Rate):** The operating point where FPR = FNR. Lower is better.
- **FPR@5%FNR:** The false positive rate when the classifier is tuned to accept at most 5% false negatives. Operationally relevant for security applications where missed steganography is most costly.

```python
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from typing import Dict


def compute_detection_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Dict[str, float]:
    """
    Compute all detection metrics from true labels and predicted probabilities.

    Args:
        y_true:  1-D array of true labels (0=cover, 1=stego).
        y_score: 1-D array of predicted stego probabilities in [0, 1].

    Returns:
        Dictionary of metric names to values.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    # Youden's J: optimal threshold
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    y_pred = (y_score >= best_threshold).astype(int)
    accuracy = (y_pred == y_true).mean()

    # EER: find where FPR ≈ FNR
    fnr = 1.0 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0

    # FPR @ 5% FNR
    target_fnr = 0.05
    valid = fnr <= target_fnr
    fpr_at_5fnr = fpr[valid].min() if valid.any() else 1.0

    return {
        "AUC": float(auc),
        "Accuracy_Youden": float(accuracy),
        "EER": float(eer),
        "FPR_at_5pct_FNR": float(fpr_at_5fnr),
        "Optimal_Threshold": float(best_threshold),
    }
```

#### 6.2 Image Quality Metrics

**PSNR (Peak Signal-to-Noise Ratio):** Target > 40 dB. Measures pixel-level fidelity.

$$\text{PSNR} = 10 \log_{10}\!\left(\frac{255^2}{\text{MSE}}\right)$$

where $\text{MSE} = \frac{1}{HWC}\sum_{i,j,c}(x_{i,j,c} - x'_{i,j,c})^2$.

**SSIM (Structural Similarity Index):** Target > 0.95. Combines luminance ($\mu$), contrast ($\sigma$), and structure ($\sigma_{xy}$) comparisons into a single perceptual similarity score. More closely aligned with human perception than PSNR.

**FSIM (Feature Similarity):** Based on phase congruency (a biologically plausible feature detector) and gradient magnitude. Robust to local distortions that SSIM may miss.

```python
import torch
import piq
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim_sk
from typing import Dict


def compute_image_quality(
    cover_path: str,
    stego_path: str,
) -> Dict[str, float]:
    """
    Compute PSNR, SSIM, and FSIM between cover and stego images.
    """
    cover = np.array(Image.open(cover_path).convert("RGB"), dtype=np.float32)
    stego = np.array(Image.open(stego_path).convert("RGB"), dtype=np.float32)

    # PSNR
    mse = np.mean((cover - stego) ** 2)
    psnr = 10.0 * np.log10(255.0 ** 2 / mse) if mse > 0 else float("inf")

    # SSIM (skimage, channel-wise average)
    ssim_val = ssim_sk(cover, stego, channel_axis=2, data_range=255.0)

    # FSIM (piq)
    cover_t = torch.from_numpy(cover / 255.0).permute(2, 0, 1).unsqueeze(0)
    stego_t = torch.from_numpy(stego / 255.0).permute(2, 0, 1).unsqueeze(0)
    fsim_val = piq.fsim(cover_t, stego_t, data_range=1.0).item()

    return {
        "PSNR_dB": float(psnr),
        "SSIM": float(ssim_val),
        "FSIM": float(fsim_val),
    }
```

#### 6.3 Statistical Analysis

- **Two-way ANOVA:** Carrier source (real vs. ML) × embedding method (LSB vs. DCT) on AUC scores, with payload level as a continuous covariate. This tests for main effects and interaction effects jointly.
- **Pairwise comparisons:** Paired t-test (if AUC distributions are normal by Shapiro-Wilk) or Wilcoxon signed-rank test (nonparametric fallback) comparing real vs. ML within each condition.
- **Effect sizes:** Cohen's $d = (\mu_1 - \mu_2) / \sigma_{\text{pooled}}$. Values: small = 0.2, medium = 0.5, large = 0.8.
- **Multiple comparison correction:** Bonferroni correction applied across all 6 hypotheses: adjusted $\alpha = 0.05 / 6 \approx 0.0083$.

```python
import numpy as np
import scipy.stats as stats
import statsmodels.formula.api as smf
import pandas as pd
from typing import Dict


def run_anova(results_df: pd.DataFrame) -> Dict:
    """
    Two-way ANOVA: AUC ~ carrier_source * embed_method + payload_level.

    results_df columns: auc, carrier_source (real/ml), embed_method (lsb/dct),
                        payload_level (low/medium/high).
    """
    model = smf.ols(
        "auc ~ C(carrier_source) * C(embed_method) + C(payload_level)",
        data=results_df,
    ).fit()
    anova_table = smf.ols(
        "auc ~ C(carrier_source) * C(embed_method) + C(payload_level)",
        data=results_df,
    ).fit()
    from statsmodels.stats.anova import anova_lm
    table = anova_lm(model, typ=2)
    return {"anova_table": table, "model_summary": model.summary()}


def pairwise_test(group1: np.ndarray, group2: np.ndarray) -> Dict[str, float]:
    """
    Paired Wilcoxon signed-rank test + Cohen's d between two AUC vectors.
    """
    stat, p_value = stats.wilcoxon(group1, group2)
    diff = group1 - group2
    d = diff.mean() / (diff.std(ddof=1) + 1e-9)
    return {
        "wilcoxon_stat": float(stat),
        "p_value": float(p_value),
        "p_value_bonferroni": float(min(p_value * 6, 1.0)),
        "cohen_d": float(d),
    }
```

---

### 7. Expected Results Structure

#### 7.1 Hypotheses to Test

| Hypothesis | Description | Confirming result | Disconfirming result |
|---|---|---|---|
| **H1** (Distributional Difference) | ML-generated carriers are statistically distinct from real photographs in ways that affect detectability | Condition A AUC ≠ Condition B AUC (significant by Wilcoxon, Bonferroni-corrected) | A ≈ B, no significant difference |
| **H2** (Payload Divergence) | The difference in detectability between real and ML widens as payload increases | AUC vs. payload plot shows diverging lines for real/ML | Lines remain parallel across payload levels |
| **H3** (Method Interaction) | LSB and DCT embedding interact differently with carrier origin | Significant carrier × method interaction term in ANOVA | Non-significant interaction; only main effects |
| **H4** (Cross-Domain Drop) | Classifiers trained on one carrier type perform significantly worse when tested on the other | Conditions C and D AUC 10–25% below A and B respectively | Cross-domain AUC approximately equal to within-domain AUC |
| **H5** (Asymmetric Transfer) | Transfer from real-to-ML (Condition C) differs from ML-to-real (Condition D) | C AUC ≠ D AUC (statistically significant) | C ≈ D |
| **H6** (Encryption Effect) | AES-256 encryption does not affect detectability | No significant difference between encrypted and plain AUC within the same conditions | Encrypted payloads are significantly harder or easier to detect |

#### 7.2 Results Tables to Report

1. **Main results table:** AUC (mean ± std across folds) organized as a 3-way table [carrier × method × payload] for within-domain conditions A and B, reported separately per classifier.
2. **Cross-domain table:** AUC for conditions A–E, organized by classifier (rows) and condition (columns), for a fixed representative configuration (e.g., LSB, medium payload, no encryption).
3. **Image quality table:** PSNR, SSIM, and FSIM organized by [carrier × method × payload], showing mean ± std across all 250 images in each group.
4. **Statistical table:** ANOVA F-statistics, degrees of freedom, p-values (raw and Bonferroni-corrected), and Cohen's d for all 6 hypotheses.

#### 7.3 Figures to Produce

1. **ROC curves:** Overlay conditions A–E on the same plot. Produce one subplot per classifier; another set per embedding method. Color-code by condition.
2. **Payload sensitivity curves:** Line plot of AUC vs. payload level (x-axis: Low, Medium, High), one line for real, one for ML. One subplot per detector × method combination (4 subplots total).
3. **Heatmap:** AUC matrix with carrier origin on one axis, payload level on the other, method separated into two side-by-side heatmaps. Use a diverging colormap centered at 0.75 (a meaningful mid-point between chance and near-perfect).
4. **Scatter plot:** PSNR (x-axis) vs. AUC (y-axis) for all individual stego images. Color by carrier origin, marker shape by method. This visualizes the quality–detectability tradeoff.
5. **Bar chart:** Cohen's d effect sizes (y-axis) for each hypothesis H1–H6 (x-axis). Include error bars from bootstrapped confidence intervals. Draw reference lines at d=0.2, 0.5, 0.8.

#### 7.4 Interpreting a Null Result

If H1 is **not** confirmed, i.e., there is no statistically significant difference in detectability between real and ML-generated carriers, this is a valid, important, and publishable finding. It would suggest:

- The noise statistics of modern ML-generated images (SD v2.1, StyleGAN3) have converged sufficiently to natural photographs that their steganographic carrier properties are indistinguishable.
- Steganalysis tools are sensitive to embedding artifacts rather than carrier statistics, confirming their domain-agnostic nature.
- Practitioners do not need to retrain or adapt steganalysis pipelines for ML-generated content.

A null result should be framed around its **statistical power**: report the 95% confidence interval on the difference in AUC and the effect size bound within which H1 can be ruled out (e.g., "We can exclude effect sizes larger than d=0.15 with 90% power"). This demonstrates rigor and provides actionable quantitative bounds.

---

### 8. Division of Labor & Timeline

**Team structure:**
- **Data team (members 1–2):** Phase 1, dataset construction and quality control.
- **Stego team (members 3–4):** Phase 2, embedding pipeline implementation and stego image generation.
- **Classification team (members 5–6):** Phases 3–5, classifier implementation, training, evaluation, and statistical analysis.

| Week | Data Team (1–2) | Stego Team (3–4) | Classification Team (5–6) |
|---|---|---|---|
| 1 | Set up download scripts; begin RAISE/COCO/Flickr30k download | Set up repository structure; implement LSB embedding | Implement RS Analysis detector; implement chi-square attack |
| 2 | Complete real image download; run normalization pipeline | Implement DCT embedding | Implement SRM feature extractor + FLD training loop; implement evaluation pipeline |
| 3 | BRISQUE quality gate; regenerate failed images | Generate all LSB stego variants (3 payload levels) | Run RS Analysis + chi-square on all cover images (baseline) |
| 4 | Generate SD v2.1 images (250); run quality gate | Generate all DCT stego variants; verify directory structure | Run RS Analysis + chi-square on all stego images; begin SRM training (Conditions A, B) |
| 5 | Generate StyleGAN3 images (250); finalize matching | Full dataset integrity check; pair-verification script | Complete SRM runs (Conditions C, D, E); run statistical analysis |
| 6 | Dataset documentation; README per data directory | Support detection team as needed | Produce all figures; draft results section |
| 7 | Full team: internal review, writing, presentation preparation | | |

---

### 9. References

[1] J. Fridrich, M. Goljan, and R. Du, "Detecting LSB Steganography in Color and Grayscale Images," *IEEE Multimedia*, vol. 8, no. 4, pp. 22–28, Oct.–Dec. 2001. *(RS Analysis)*

[2] A. Westfeld and A. Pfitzmann, "Attacks on Steganographic Systems," in *Proc. Information Hiding: 3rd Int. Workshop*, Dresden, Germany, Oct. 1999, pp. 61–76. Lecture Notes in Computer Science, vol. 1768. *(Chi-square attack)*

[3] J. Fridrich and J. Kodovský, "Rich Models for Steganalysis of Digital Images," *IEEE Transactions on Information Forensics and Security*, vol. 7, no. 3, pp. 868–882, Jun. 2012. *(SRM)*

[4] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, "High-Resolution Image Synthesis with Latent Diffusion Models," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, New Orleans, LA, USA, 2022, pp. 10684–10695.

[5] T. Karras, M. Laine, M. Aittala, J. Hellsten, J. Lehtinen, and T. Aila, "Alias-Free Generative Adversarial Networks," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 34, 2021, pp. 852–863.

[6] D.-T. Dang-Nguyen, C. Pasquini, V. Conotter, and G. Boato, "RAISE: A Raw Images Dataset for Digital Image Forensics," in *Proc. ACM Multimedia Systems Conf. (MMSys)*, Portland, OR, USA, 2015, pp. 219–224.

[7] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, and C. L. Zitnick, "Microsoft COCO: Common Objects in Context," in *Proc. European Conf. Computer Vision (ECCV)*, Zurich, Switzerland, 2014, pp. 740–755.

[8] P. Young, A. Lai, M. Hodosh, and J. Hockenmaier, "From Image Descriptions to Visual Denotations: New Similarity Metrics for Semantic Inference over Event Descriptions," *Transactions of the Association for Computational Linguistics*, vol. 2, pp. 67–78, 2014.

[9] B. Chen and G. W. Wornell, "Quantization Index Modulation: A Class of Provably Good Methods for Digital Watermarking and Information Embedding," *IEEE Transactions on Information Theory*, vol. 47, no. 4, pp. 1423–1443, May 2001.

[10] A. Westfeld, "F5 — A Steganographic Algorithm," in *Proc. Information Hiding: 4th Int. Workshop*, Pittsburgh, PA, USA, Apr. 2001, pp. 289–302. Lecture Notes in Computer Science, vol. 2137. Berlin: Springer.

[11] S.-Y. Wang, O. Wang, R. Zhang, A. Owens, and A. A. Efros, "CNN-Generated Images Are Surprisingly Easy to Spot… For Now," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, Seattle, WA, USA, 2020, pp. 8695–8704.
