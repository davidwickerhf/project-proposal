# Does the Source of Carrier Image Affect Steganographic Detectability? A Comparative Study of Real vs. ML-Generated Image Steganography

**Course:** [Course Name — Project 2.2]
**Authors:** Nico, Nikolas, Abdul, Daria, Jimena, David
**Date:** February 2026
**Status:** Draft for Team Review

---

## 1. Abstract

This proposal outlines a comparative study investigating whether the origin of a carrier image — real (human-photographed) versus machine-learning-generated — affects the detectability of steganographic content embedded using identical methods and payload sizes. As AI-generated imagery becomes increasingly prevalent and indistinguishable from photographs, understanding how its statistical properties interact with steganographic embedding is both timely and practically significant. We propose a controlled experimental framework that pairs matched datasets of real and ML-generated images, applies standardized embedding techniques (LSB and DCT-based) at varying payload rates with and without payload encryption, and evaluates detectability using classical steganalysis detectors — a training-free statistical approach (RS Analysis / chi-square attack) and a classical-ML approach (SRM + FLD ensemble) — including cross-domain generalization tests. Image quality is measured throughout as a secondary outcome. The findings would contribute to both the steganography and AI-image forensics communities by clarifying whether synthetic imagery offers inherent advantages or vulnerabilities as a steganographic medium.

---

## 2. Introduction and Motivation

Image steganography — the practice of concealing secret data within digital images — has been studied extensively over the past two decades. Techniques range from simple Least Significant Bit (LSB) substitution in the spatial domain to sophisticated frequency-domain methods based on the Discrete Cosine Transform (DCT), as well as transform-domain approaches including DWT and content-adaptive methods [1, 2, 3]. The security of these methods depends heavily on the statistical properties of the carrier image: embedding alters the carrier's distribution in subtle ways, and steganalysis classifiers exploit these distributional shifts to detect hidden content [4, 5].

Simultaneously, a revolution has occurred in image generation. Models such as Stable Diffusion [6], DALL-E 3 [7], Midjourney [8], and StyleGAN3 [9] now produce images that are perceptually indistinguishable from real photographs in many settings. These models synthesize images through fundamentally different generative processes — latent diffusion, transformer-based token prediction, GAN-based adversarial training — each imposing its own statistical fingerprint on the output image. This raises a natural and largely unexplored question:

> **Does the generative origin of a carrier image affect how detectable steganographic embedding becomes?**

This question matters for several reasons. First, from a **security perspective**, if ML-generated images prove harder to steganalyze, adversaries could exploit synthetic carriers to evade detection — a concern for law enforcement and digital forensics. Second, from a **theoretical perspective**, the interaction between a generative model's learned distribution and steganographic distortion is poorly understood. Third, the **practical implications** are significant: as AI-generated images proliferate in social media, news, and digital communications, the steganographic attack surface is changing in ways the research community has not yet fully characterized.

Some preliminary work in the image domain has begun to explore whether AI-generated images behave differently as steganographic carriers [10, 11], but no systematic, controlled study comparing real vs. ML-generated images under identical embedding conditions has been conducted. This proposal aims to fill that gap.

---

## 3. Research Questions

**Primary Research Question (RQ):**
Does the source of the carrier image (real human-photographed vs. ML-generated) affect the detectability of image steganography when using identical embedding methods and payload sizes?

**Secondary Research Questions:**

- **RQ1 (Payload Sensitivity):** How does payload size influence detectability across real and ML-generated images? Do the two carrier types diverge in detectability at different embedding rates?

- **RQ2 (Embedding Method Interaction):** Do different embedding methods — specifically spatial-domain LSB substitution versus frequency-domain DCT-based embedding — behave differently depending on the carrier image's origin?

- **RQ3 (Encryption Effect):** Does encrypting the payload before embedding affect detectability, and does this interaction vary across carrier types?

- **RQ4 (Cross-Domain Generalization):** How well do steganalysis classifiers generalize across carrier domains? That is, if a classifier is trained on real images (cover + stego pairs), how does it perform when tested on ML-generated images, and vice versa?

---

## 4. Background and Previous Work

### 4.1 Image Steganography Techniques

Image steganography methods are generally categorized by the domain in which embedding occurs.

**Spatial-domain methods** operate directly on pixel values. The most fundamental is **LSB substitution**, which replaces the least significant bits of pixel color channel values with message bits. While offering high capacity, LSB methods are vulnerable to statistical attacks that detect distributional asymmetries introduced by bit replacement [12, 13]. Enhancements include pixel-value differencing (PVD) coding [14] and adaptive embedding strategies that target textured regions [15].

**Frequency-domain methods** embed data in transform coefficients. **DCT-based steganography** modifies discrete cosine transform coefficients of image blocks, analogous to JPEG steganography. These methods can exploit perceptual masking to embed data in visually insignificant coefficients [16, 17]. **DWT-based methods** embed data in wavelet coefficients, offering multi-resolution analysis [18]. **Spread spectrum** techniques distribute the hidden message across the frequency spectrum, offering robustness at the cost of capacity [19].

**Content-adaptive methods** such as WOW, HILL, and S-UNIWARD [20] select embedding locations based on local image complexity, minimizing the statistical footprint of embedding.

For this study, we focus on **LSB substitution** (as a representative spatial-domain method) and **DCT-based embedding** (as a representative frequency-domain method), as these represent the two dominant paradigms and allow clean comparison of domain-specific effects.

### 4.2 Image Steganalysis

Steganalysis — the detection of steganographic content — has evolved from hand-crafted statistical features to deep learning approaches.

**Traditional approaches** extract features such as Rich Models (SRM), co-occurrence matrices, and higher-order statistics, then classify using ensemble classifiers [4]. Fridrich and Kodovský [4] introduced the Spatial Rich Model, which remains a strong baseline.

**Training-free statistical methods** exploit distributional properties introduced by embedding. The **chi-square attack** [23] (Westfeld & Pfitzmann, 1999) tests whether pixel value pairs (2k, 2k+1) have equalised frequencies — the statistical signature of LSB substitution. **RS Analysis** [24] (Fridrich, Goljan & Du, 2001) analyses pixel group regularity to estimate the embedding rate analytically, generalising beyond the chi-square setting.

**Deep learning approaches** have significantly advanced the field in recent years, with CNN and residual network architectures achieving near-perfect accuracy on standard benchmarks [25, 26]. A comprehensive survey by Luo et al. [27] reviews these developments, noting that image steganalysis is the most mature subfield.

### 4.3 ML-Generated Images

Modern image generation has achieved remarkable quality across several paradigms.

**Latent diffusion models:** Stable Diffusion [6] applies diffusion processes in a compressed latent space using a VAE, producing highly photorealistic images from text prompts. It has become the dominant open-source paradigm.

**Transformer-based models:** DALL-E 3 [7] uses CLIP-guided token prediction to generate semantically rich, highly controllable images. Midjourney [8] similarly produces high-quality artistic and photorealistic outputs.

**GAN-based models:** StyleGAN3 [9] generates high-resolution, diverse face and scene images through adversarial training. GANs are particularly well-characterized statistically and have been widely studied in the deepfake detection literature.

Each generative paradigm imposes distinct statistical signatures on its output — diffusion models produce particular noise residual patterns, GANs produce spectral artifacts in high frequencies, and transformer models yield specific spatial coherence patterns. These signatures are already exploited by **image deepfake/forgery detection** systems [28, 29, 30], but their interaction with steganographic embedding has not been studied systematically.

### 4.4 Steganography in AI-Generated Media

The closest existing work falls into several related areas:

**Generative steganography** uses GANs or other generative models to directly produce stego-content, rather than modifying a pre-existing cover signal. Hu et al. [31] and Liu et al. [32] proposed GAN-based coverless steganography methods. These approaches differ fundamentally from our proposal: they use generation *as* the embedding mechanism, whereas we investigate using ML-generated images *as a passive carrier* for traditional embedding methods.

**Coverless steganography** generates synthetic content that inherently encodes a message without any cover modification [33, 34]. While related, this is a different threat model from our study.

**AI-generated images as carriers:** De et al. [10] investigated steganographic secret sharing via AI-generated photorealistic images, finding that minimum-entropy coupling can achieve statistically undetectable embedding. This is the closest work to our proposal, but it does not systematically compare real vs. ML-generated images under controlled, identical embedding conditions.

**Steganography beyond space-time:** Very recent work [35] explored embedding messages through multimodal AI pipelines, pointing toward the convergence of generative AI and information hiding.

### 4.5 Cross-Domain Generalization in Steganalysis

Cross-domain generalization is a well-documented challenge. Steganalysis models trained on one domain (e.g., images from a specific camera or processing pipeline) suffer significant performance degradation when tested on a different domain [36]. Recent approaches address this through unsupervised domain adaptation [37] and self-supervised feature learning [38]. However, the specific domain shift from real photographs to ML-generated images has not been studied in the steganalysis context — a gap our study directly addresses.

### 4.6 Summary of Research Gap

The literature reveals a clear gap at the intersection of three active research areas: (i) image steganography/steganalysis, (ii) ML-generated image characterization, and (iii) cross-domain robustness of detection systems. While individual components are well-studied, **no existing work systematically compares real vs. ML-generated images as steganographic carriers under controlled conditions.** Our study proposes to bridge this gap with a rigorous experimental framework.

---

## 5. Experimental Design and Methodology

### 5.1 Overview

The experiment follows a **2 × 2 × 3 × 2 factorial design** with three primary factors:

| Factor | Levels |
|---|---|
| **Carrier source** | Real (human-photographed) vs. ML-generated |
| **Embedding method** | Spatial (LSB) vs. Frequency-domain (DCT) |
| **Payload rate** | Low, Medium, High (see §5.4) |
| **Detector** | RS Analysis (training-free) vs. SRM (classical ML) |

For each combination of factors, we produce matched cover–stego pairs and evaluate detectability using both detectors across five train/test conditions, including cross-domain transfer experiments.

### 5.2 Dataset Construction

#### Real Image Dataset
We will sample from established photographic datasets to construct the real image dataset:

- **RAISE** [39]: 8,156 high-quality RAW images from DSLR cameras, covering diverse scenes and subjects.
- **COCO** [40]: Common Objects in Context — a large-scale dataset of natural photographs with diverse content.
- **Flickr30k** [41]: 31,000 images collected from Flickr, covering a wide range of everyday scenes.

We will select **500 images** (balanced across datasets and categories), each normalized to:
- Dimensions: **512 × 512 pixels**
- Color space: **RGB**
- Bit depth: **8-bit per channel**
- Format: **PNG (lossless)**

Specifically: RAISE (250 images), COCO (150 images), Flickr30k (100 images). The expanded set of 500 real images improves statistical power for cross-domain experiments and reduces variance in ANOVA comparisons.

#### ML-Generated Image Dataset
We will generate a matched set of **500 images** using representative generative models:

1. **Stable Diffusion v2.1** [6]: Latent diffusion model producing photorealistic images from text prompts.
2. **StyleGAN3** [9]: GAN-based model for high-resolution image synthesis with well-characterized spectral properties.

Generated images will be normalized to identical format specifications (512×512, RGB, 8-bit, PNG). This yields approximately **250 images per generative model**, enabling sub-analysis by generation architecture.

#### Matching Protocol
To ensure fair comparison, we control for:
- **Content**: ML-generated images use the same descriptive prompts as the semantic content of real image samples (e.g., "a dog in a park", "city street at night").
- **Dimensions and format**: Identical across all samples.
- **Luminance**: All images normalized to consistent mean luminance.

### 5.3 Embedding Methods

We implement two canonical steganography methods:

**LSB Substitution (Spatial Domain):**
Replace the *k* least significant bits of each 8-bit pixel channel value with message bits. We use k = 1 (standard LSB) as the baseline and k = 2, 4 for higher payload experiments. Implementation uses pseudorandom pixel selection keyed by a shared secret, applied across all three RGB channels.

**DCT-Based Embedding (Frequency Domain):**
Segment the image into non-overlapping 8×8 pixel blocks, compute the DCT of each block (per channel), and embed message bits by quantizing selected mid-frequency DCT coefficients. This approach mirrors JPEG-domain steganography (F5 / JSteg) [16, 17]. Coefficient selection prioritizes mid-frequency positions using the standard JPEG zigzag scan order, targeting visually insignificant regions.

For both methods, the embedded message consists of **pseudorandom bitstreams** (to eliminate content-dependent effects), with an optional **AES-256 encryption** pass applied to the bitstream before embedding (addressing RQ3). The same message and embedding key are used across all carrier types for a given payload rate.

### 5.4 Payload Rates

We test embedding at multiple payload rates to capture the capacity–detectability trade-off:

| Level | LSB Variant | DCT Variant | Approximate bpp |
|---|---|---|---|
| Low | k=1, 25% pixels | 10% coefficients | ~0.08 bpp |
| Medium | k=1, 50% pixels | 25% coefficients | ~0.16 bpp |
| High | k=2, 50% pixels | 50% coefficients | ~0.32 bpp |

These three rates span from conservative (hard to detect, low capacity) to aggressive (easy to detect, high capacity), allowing us to trace the detection curve for each carrier type without excessive experimental overhead.

### 5.5 Steganalysis Detectors

We use two detectors deliberately chosen to stay within the scope of classical signal processing and statistics, avoiding deep learning entirely.

1. **RS Analysis** [24]: A fully training-free statistical test (Fridrich, Goljan & Du, 2001). Analyses pixel group regularity to estimate the embedding rate $\hat{p}$ analytically. Requires no training data and generalizes across image domains by construction — any difference in detection rate between real and ML-generated images reflects only the carriers' statistical properties, not classifier bias.

2. **SRM + FLD Ensemble** [4]: Spatial Rich Model (Fridrich & Kodovský, 2012). Extracts high-pass residual co-occurrence features (~35,000 dimensions) and classifies with an ensemble of Fisher Linear Discriminants — a classical statistical classifier, not a neural network. Requires labeled cover/stego training pairs but runs on CPU in minutes. Included because it handles DCT-domain embedding better than training-free methods, and its hand-crafted features are expected to generalize across the real/ML domain boundary better than learned representations.

The chi-square attack (Westfeld & Pfitzmann, 1999) [23] is also applied as a supplementary analytical check on LSB results. It tests whether pixel value pairs (2k, 2k+1) have equalised frequencies — the statistical signature of LSB substitution.

Detectors are evaluated as binary classifiers (cover vs. stego). The training-free methods produce a continuous score per image (embedding rate estimate or p-value); SRM is trained with 3-fold cross-validation.

### 5.6 Experimental Conditions

We define four training–testing configurations to address our research questions:

| Condition | Training Data | Testing Data | Purpose |
|---|---|---|---|
| **A (Real→Real)** | Real cover + stego | Real cover + stego | Baseline: standard steganalysis |
| **B (ML→ML)** | ML-generated cover + stego | ML-generated cover + stego | Within-domain ML performance |
| **C (Real→ML)** | Real cover + stego | ML-generated cover + stego | Cross-domain: does training on real generalize to ML? |
| **D (ML→Real)** | ML-generated cover + stego | Real cover + stego | Cross-domain: does training on ML generalize to real? |

An additional **mixed-domain** condition trains on a 50/50 blend of real and ML-generated data and tests on each domain separately.

### 5.7 Evaluation Metrics

**Detection Performance:**
- **ROC-AUC**: Primary metric — threshold-independent measure of classifier discrimination.
- **Accuracy** at optimal threshold (Youden's J statistic).
- **Equal Error Rate (EER)**: Threshold where false positive rate equals false negative rate.
- **False Positive Rate at 5% False Negative Rate (FPR@5%FNR)**: Operationally relevant metric for high-security scenarios.

**Image Quality (Imperceptibility):**
- **PSNR** (Peak Signal-to-Noise Ratio): Standard pixel-level distortion measure (dB).
- **SSIM**: Structural Similarity Index, capturing luminance, contrast, and structural changes.
- **FSIM**: Feature Similarity Index, based on phase congruency and gradient magnitude.

**Payload Integrity:**
- **Bit Error Rate (BER)**: Fraction of embedded bits incorrectly extracted, verifying embedding correctness.

### 5.8 Statistical Analysis

All comparisons will use appropriate statistical tests:
- **Two-way ANOVA** (carrier type × embedding method) on AUC scores, with payload rate as a covariate.
- **Paired t-tests or Wilcoxon signed-rank tests** for pairwise comparisons (real vs. ML-generated) within each condition.
- **Effect sizes** (Cohen's d) reported alongside p-values.
- **Bonferroni correction** for multiple comparisons.

---

## 6. Expected Results and Hypotheses

Based on the literature and the known properties of ML-generated images, we formulate the following hypotheses:

**H1 (Distributional Difference):** ML-generated images will exhibit different steganalysis detectability compared to real images, even under identical embedding. This is because generative models impose learned statistical regularities (e.g., smoother textures, spectral artifacts at high frequencies in GANs, specific noise patterns in diffusion models) that interact with embedding distortion differently than the natural statistical structure of photographs.

**H2 (Payload Sensitivity):** The divergence in detectability between real and ML-generated carriers will increase with payload size. At low embedding rates, both carrier types may be similarly difficult to steganalyze; at high rates, the differing noise floors and distributional properties will become more consequential.

**H3 (Method Interaction):** Frequency-domain (DCT) embedding will show greater sensitivity to carrier origin than spatial-domain (LSB) embedding. DCT methods modify coefficients whose distribution is more directly shaped by the generative process, whereas LSB modifications affect individual pixel bits, which may be more uniformly distributed across carrier types.

**H4 (Cross-Domain Degradation):** Steganalysis classifiers will suffer significant accuracy drops in cross-domain conditions (C and D), with performance degradation of **10–25% in AUC** compared to within-domain conditions (A and B). This would parallel findings in image deepfake detection and domain transfer steganalysis [36].

**H5 (Asymmetric Transfer):** Transfer from Real→ML (condition C) will perform differently than ML→Real (condition D). We hypothesize that classifiers trained on real photographs may perform worse on ML-generated images, because ML-generated images' distinctive statistical profile (e.g., GAN spectral peaks, diffusion noise patterns) could mask or alter the embedding artifacts the classifier learned to detect in natural images.

**H6 (Encryption Effect):** Encrypting the payload before embedding will not significantly affect steganalysis detectability, since the randomness of the payload is similar whether pseudorandom or AES-encrypted. However, any measurable difference would provide evidence that payload structure contributes to detection — an underexplored question.

---

## 7. Tools and Implementation

### 7.1 Image Generation Pipeline
- **Stable Diffusion**: Via `diffusers` library (HuggingFace), run locally
- **StyleGAN3**: Official NVIDIA implementation (PyTorch)

### 7.2 Steganography Implementation
- **LSB embedding**: Custom Python implementation using NumPy/Pillow for direct pixel manipulation
- **DCT embedding**: Custom implementation using `scipy.fft.dctn` on 8×8 image blocks
- **AES encryption**: `cryptography` Python library (AES-256-CBC)
- Reference implementations: OpenStego [42], Steghide [43]

### 7.3 Steganalysis Detectors
- **RS Analysis & chi-square attack**: Custom Python implementation using NumPy (pixel group regularity computation) and `scipy.stats.chi2` (chi-square p-values)
- **SRM feature extraction**: Custom NumPy implementation of high-pass residual filters + co-occurrence histograms; `scikit-learn` SGDClassifier for FLD ensemble
- **No deep learning frameworks required**: all detectors run on CPU with standard scientific Python libraries

### 7.4 Evaluation
- **PSNR/SSIM/FSIM**: `scikit-image` and `piq` Python libraries
- **Statistical analysis**: SciPy, statsmodels
- **Visualization**: Matplotlib, Seaborn

---

## 8. Proposed Timeline

| Week | Activity |
|---|---|
| 1 | Dataset collection (real images) and generation (ML images); format normalization and quality screening |
| 2 | Steganography implementation and verification (LSB + DCT); embedding at all payload rates, with and without encryption |
| 3–4 | Steganalysis classifier implementation and training (Conditions A and B) |
| 5 | Cross-domain experiments (Conditions C and D) and mixed-domain training |
| 6 | Statistical analysis, visualization, and interpretation |
| 7 | Paper writing, revision, and submission |

---

## 9. Ethical Considerations

This research involves no human subjects. All real image datasets (RAISE, COCO, Flickr30k) are publicly available under their respective licenses and commonly used for research. ML-generated images are produced using open-source models (Stable Diffusion) and publicly available architectures (StyleGAN3). The steganographic methods studied are well-known; our contribution is analytical rather than the development of new evasion techniques. We will discuss the dual-use implications of our findings — specifically, the possibility that adversaries could exploit ML-generated carriers — in the paper's discussion section.

---

## 10. Expected Contributions

1. **First systematic comparison** of real vs. ML-generated images as steganographic carriers under controlled experimental conditions.
2. **Quantitative evidence** on whether synthetic imagery offers steganographic advantages or vulnerabilities compared to natural photographs.
3. **Cross-domain generalization analysis** for image steganalysis, extending domain adaptation research to the real-vs-synthetic boundary.
4. **Encryption interaction analysis**: whether payload encryption affects detectability — a secondary but novel contribution.
5. **Practical recommendations** for steganalysis practitioners on whether existing detectors need retraining or adaptation for the growing prevalence of ML-generated images.

---

## References

[1] Petitcolas, F. A. P., Anderson, R. J., & Kuhn, M. G. (1999). Information Hiding — A Survey. *Proceedings of the IEEE*, 87(7), 1062–1078.

[2] Cheddad, A., Condell, J., Curran, K., & McKevitt, P. (2010). Digital Image Steganography: Survey and Analysis of Current Methods. *Signal Processing*, 90(3), 727–752.

[3] Hussain, M., Wahab, A. W. A., Idris, Y. I. B., Ho, A. T. S., & Jung, K. H. (2018). Image Steganography in Spatial Domain: A Survey. *Signal Processing: Image Communication*, 65, 46–66.

[4] Fridrich, J. & Kodovský, J. (2012). Rich Models for Steganalysis of Digital Images. *IEEE Transactions on Information Forensics and Security*, 7(3), 868–882.

[5] Luo, Y., et al. (2024). Deep Learning for Steganalysis of Diverse Data Types: A Review. *Neurocomputing*.

[6] Rombach, R., et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. In *Proc. IEEE CVPR 2022*.

[7] OpenAI. (2023). DALL-E 3: Improving Image Generation with Better Captions. *Technical Report*.

[8] Midjourney. (2023). Midjourney V6. *Midjourney Blog*.

[9] Karras, T., et al. (2021). Alias-Free Generative Adversarial Networks (StyleGAN3). In *Proc. NeurIPS 2021*.

[10] De, A., Kinzel, W., & Kanter, I. (2022). Steganographic Secret Sharing via AI-Generated Photorealistic Images. *Journal on Wireless Communications and Networking*, Springer.

[11] Li, J., et al. (2021). Disguise of Steganography Behaviour: Steganography Using Image Processing with Generative Adversarial Network. *Security and Communication Networks*, Wiley.

[12] Fridrich, J., Goljan, M., & Du, R. (2001). Detecting LSB Steganography in Color and Grayscale Images. *IEEE Multimedia*, 8(4), 22–28.

[13] Chan, C. K., & Cheng, L. M. (2004). Hiding Data in Images by Simple LSB Substitution. *Pattern Recognition*, 37(3), 469–474.

[14] Wu, D. C., & Tsai, W. H. (2003). A Steganographic Method for Images by Pixel-Value Differencing. *Pattern Recognition Letters*, 24(9–10), 1613–1626.

[15] Holub, V., Fridrich, J., & Denemark, T. (2014). Universal Distortion Function for Steganography in an Arbitrary Domain. *EURASIP Journal on Information Security*.

[16] Provos, N., & Honeyman, P. (2003). Hide and Seek: An Introduction to Steganography. *IEEE Security & Privacy*, 1(3), 32–44.

[17] Westfeld, A. (2001). F5 — A Steganographic Algorithm: High Capacity Despite Better Steganalysis. In *Proc. Information Hiding Workshop*, Springer.

[18] Cvejic, N., & Seppanen, T. (2004). Increasing Capacity of LSB-Based Image Steganography. In *Proc. ICICS 2004*.

[19] Marvel, L. M., Boncelet, C. G., & Retter, C. T. (1999). Spread Spectrum Image Steganography. *IEEE Transactions on Image Processing*, 8(8), 1075–1083.

[20] Holub, V., & Fridrich, J. (2012). Designing Steganographic Distortion Using Directional Filters. In *Proc. IEEE WIFS 2012*.

[21] Chen, B. & Wornell, G. W. (2001). Quantization Index Modulation: A Class of Provably Good Methods for Digital Watermarking and Information Embedding. *IEEE Trans. Information Theory*, 47(4), 1423–1443.

[22] Maleki, N., et al. (2012). Multibit Quantization Index Modulation: A High-Rate Robust Data-Hiding Method. *Journal of King Saud University - Computer and Information Sciences*.

[23] Westfeld, A., & Pfitzmann, A. (1999). Attacks on Steganographic Systems. In *Proc. 3rd International Workshop on Information Hiding*, Springer LNCS 1768, pp. 61–76.

[24] Fridrich, J., Goljan, M., & Du, R. (2001). Reliable Detection of LSB Steganography in Color and Grayscale Images. In *Proc. ACM Multimedia Workshop on Multimedia and Security*, pp. 27–30.

[25] Xu, G., Wu, H. Z., & Shi, Y. Q. (2016). Structural Design of Convolutional Neural Networks for Steganalysis. *IEEE Signal Processing Letters*, 23(5), 708–712.

[26] Yousfi, Y., & Fridrich, J. (2020). An Intriguing Struggle of CNNs in JPEG Steganalysis. *IEEE Signal Processing Letters*, 27, 1691–1695.

[27] Luo, Y., et al. (2024). Deep Learning for Steganalysis of Diverse Data Types: A Review of Methods, Taxonomy, Challenges and Future Directions. *Neurocomputing*, Elsevier.

[28] Wang, S. Y., Wang, O., Zhang, R., Owens, A., & Efros, A. A. (2020). CNN-Generated Images Are Surprisingly Easy to Spot... For Now. In *Proc. IEEE CVPR 2020*.

[29] Gragnaniello, D., et al. (2021). Are GAN Generated Images Easy to Detect? A Critical Analysis of the State-of-the-Art. In *Proc. IEEE ICME 2021*.

[30] Corvi, R., et al. (2023). On the Detection of Synthetic Images Generated by Diffusion Models. In *Proc. IEEE ICASSP 2023*.

[31] Hu, P., et al. (2023). A Coverless Image Steganography Based on Generative Adversarial Networks. *MDPI Electronics*.

[32] Liu, X., et al. (2024). Message-Driven Generative Steganography Using GAN. *IEEE TDSC*.

[33] Duan, X., et al. (2020). Coverless Steganography Based on Generative Adversarial Network. *EURASIP Journal on Image and Video Processing*.

[34] Liu, J., et al. (2017). Coverless Information Hiding Based on Generative Model. *arXiv:1712.06951*.

[35] Steganography Beyond Space-Time with Chain of Multimodal AI. (2025). *Scientific Reports*.

[36] Cross-Domain Steganalysis. (2024). Various venues.

[37] Unsupervised Domain Adaptation for Steganalysis. (2025). *Applied Soft Computing*, Elsevier.

[38] Self-Supervised Learning for Domain-Invariant Image Feature Representation. (2024). Various venues.

[39] Dang-Nguyen, D. T., et al. (2015). RAISE: A Raw Images Dataset for Digital Image Forensics. In *Proc. ACM MMSys 2015*.

[40] Lin, T. Y., et al. (2014). Microsoft COCO: Common Objects in Context. In *Proc. ECCV 2014*.

[41] Young, P., et al. (2014). From Image Descriptions to Visual Denotations: New Similarity Metrics for Semantic Inference over Event Descriptions. *Transactions of the Association for Computational Linguistics*, 2, 67–78.

[42] OpenStego. *Open Source Image Steganography Tool*. https://www.openstego.com

[43] Hetzl, S. (2003). Steghide: A Steganography Program. *Open Source*.

[44] Boroumand, M., Chen, M., & Fridrich, J. (2019). Deep Residual Network for Steganalysis of Digital Images. *IEEE Transactions on Information Forensics and Security*, 14(5), 1181–1193.

---

*This proposal is a working draft prepared for internal team discussion. Feedback on scope, feasibility, and research direction is welcome before finalizing the project plan.*
