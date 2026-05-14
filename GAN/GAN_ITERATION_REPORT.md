# GAN Iterative Development Report
## Deep and Reinforcement Learning Project — FEUP/FCUP 2025/2026
### Generative Model: DCGAN for Fake Face Synthesis

---

## Table of Contents

1. [Project Context](#1-project-context)
2. [Evaluation Metrics](#2-evaluation-metrics)
3. [Fixed Architecture & Shared Configuration](#3-fixed-architecture--shared-configuration)
4. [g0 — Baseline DCGAN (Instrumented)](#4-g0--baseline-dcgan-instrumented)
5. [g1 — More Data (Folds 0–20)](#5-g1--more-data-folds-020)
6. [g2 — Discriminator Balancing (Negative Result)](#6-g2--discriminator-balancing-negative-result)
7. [g3 — EMA on the Generator](#7-g3--ema-on-the-generator)
8. [g4 — Spectral Normalization on Discriminator (Negative Result)](#8-g4--spectral-normalization-on-discriminator-negative-result)
9. [g5 — Full Dataset + Best-FID Checkpoint](#9-g5--full-dataset--best-fid-checkpoint)
10. [g6 — Face Crop + Discriminator Regularization Bundle](#10-g6--face-crop--discriminator-regularization-bundle)
11. [g7 — Extended Training (400 Epochs, Regularized)](#11-g7--extended-training-400-epochs-regularized)
12. [g8 — Improved Discriminator (SpectralNorm + MinibatchStd)](#12-g8--improved-discriminator-spectralnorm--minibatchstd)
13. [Global Summary Table](#13-global-summary-table)

---

## 1. Project Context

This project was developed in the context of the Deep and Reinforcement Learning course (M.IA003, FEUP/FCUP, 2025/2026). The task requires training a **generative model** capable of producing synthetic face images indistinguishable from the real images present in the DeepFakeFace (DFF) dataset, which contains 30,000 real celebrity images sourced from the IMDB-WIKI dataset.

The generative approach chosen was a **Deep Convolutional Generative Adversarial Network (DCGAN)**, as introduced by Radford et al. (2015) [^1]. GANs frame image generation as a two-player adversarial game: a **Generator (G)** learns to map a random latent vector **z** sampled from a noise distribution into a synthetic image, while a **Discriminator (D)** learns to distinguish real images from generated ones. The training signal for G is the gradient of D's classification loss, which means the quality and informativeness of D's feedback is critical for G's improvement.

The core evaluation philosophy mandated by the project specification is **iterative**: every change must be motivated by an observed problem, grounded in published research, and validated by comparing measurable metrics before and after the change. This document fulfils that requirement in full.

---

## 2. Evaluation Metrics

### 2.1 Fréchet Inception Distance (FID)

FID, introduced by Heusel et al. (2017) [^2], is the **primary quantitative metric** used throughout this project. It computes the Fréchet distance between two multivariate Gaussians fit to the deep feature activations (layer 2048 of Inception-v3) extracted from real and generated images. Lower FID indicates that the generated distribution is statistically closer to the real one.

FID is preferred over simpler pixel-level metrics because it captures both the **quality** and **diversity** of generated images simultaneously. A generator that memorises a single real image would score perfectly on a per-pixel metric, but would have a high FID because of its lack of diversity. Conversely, a generator that produces diverse but blurry images would also score poorly, because the Gaussian fit would be wide and offset from the real distribution.

**Practical note**: all FID values in this project were computed using `torchmetrics.image.fid.FrechetInceptionDistance` with `feature=2048`, evaluated on a subset of `FID_NUM_IMAGES = 300` images per evaluation point due to memory constraints. This makes absolute FID numbers smaller-sample estimates, but they remain **consistent and comparable across all runs** since the sample size was held constant.

### 2.2 Kernel Inception Distance (KID)

KID, introduced by Bińkowski et al. (2018) [^3], is a complementary metric that estimates the **squared Maximum Mean Discrepancy (MMD)** between Inception feature distributions using a polynomial kernel. Unlike FID, KID has an unbiased estimator and does not assume Gaussianity, making it particularly reliable when the number of evaluation images is small. It is reported as `kid_mean ± kid_std`.

KID was computed alongside FID throughout all iterations. Lower KID (mean) indicates better distributional match.

### 2.3 Discriminator Output Signals

During each training epoch, three discriminator confidence statistics were tracked:

| Signal | Description | Ideal GAN Equilibrium |
|---|---|---|
| `D(x)` | Mean sigmoid probability assigned to **real** images | ~0.5 |
| `D(G(z))_fake` | Mean probability assigned to **generated** images during D's update step | ~0.5 |
| `D(G(z))_gen` | Mean probability assigned to **generated** images during G's update step | ~0.5 |

When D(x) → 1 and D(G(z)) → 0, the discriminator has saturated: it classifies all real images as real and all fakes as fake with near-certainty. In this regime, the gradient signal reaching the generator is nearly zero, because the BCEWithLogitsLoss gradient vanishes when the predicted probability is at an extreme. This phenomenon is well-documented in Goodfellow et al. (2014) [^4] and was a recurring diagnostic signal in this project.

---

## 3. Fixed Architecture & Shared Configuration

The following parameters and design choices were held constant across **all iterations** unless explicitly noted.

### 3.1 Image Size: 64×64

All images were resized and centre-cropped to **64×64 pixels** with 3 colour channels (RGB), normalised to the range `[−1, 1]` using mean and standard deviation of `(0.5, 0.5, 0.5)`.

**Justification**: The choice of 64×64 is directly motivated by the original DCGAN paper (Radford et al., 2015) [^1], which demonstrated that a 4-stage transposed-convolution generator (latent → 4×4 → 8×8 → 16×16 → 32×32 → 64×64) with matching discriminator achieves stable training and perceptually coherent outputs on face datasets such as CelebA. At higher resolutions (e.g., 128×128 or 256×256), the DCGAN architecture requires additional up/down-sampling stages and is considerably more prone to instability without further regularisation techniques such as progressive growing (Karras et al., 2018) [^5] or attention mechanisms. Given the computational constraints of this project, 64×64 was the appropriate resolution ceiling for DCGAN.

### 3.2 Batch Size: 64

**Justification**: Radford et al. (2015) [^1] originally trained DCGAN with a batch size of 128. A batch size of 64 was used here to reduce VRAM requirements while staying within the range where stochastic gradient noise provides a useful regularisation effect. BatchNorm layers, which are present in the generator (and in some discriminator variants), compute batch statistics at training time and benefit from batches large enough to estimate those statistics reliably; 64 is widely regarded as sufficient (Ioffe & Szegedy, 2015) [^6].

### 3.3 Latent Dimension: 100

The generator takes as input a vector **z** ∈ ℝ¹⁰⁰ sampled i.i.d. from 𝒩(0, 1), reshaped to `(100, 1, 1)` before the first transposed convolution. This is the exact setup from the original DCGAN paper [^1] and provides sufficient expressive capacity for a 64×64 face generator without requiring architectural modifications.

### 3.4 Generator Architecture

The generator uses an **Upsample-then-Conv** block pattern rather than the pure transposed-convolution approach from the original DCGAN paper. Specifically, nearest-neighbour upsampling is followed by a 3×3 convolution. This avoids **checkerboard artefacts** caused by uneven gradient overlap in transposed convolutions, a problem identified and analysed in Odena et al. (2016) [^7].

```
z (100×1×1)
  ↓ ConvTranspose2d(100→512, 4×4, stride=1)  → 4×4
  ↓ BatchNorm2d + ReLU
  ↓ Upsample(×2) + Conv2d(512→256, 3×3)       → 8×8
  ↓ BatchNorm2d + ReLU
  ↓ Upsample(×2) + Conv2d(256→128, 3×3)       → 16×16
  ↓ BatchNorm2d + ReLU
  ↓ Upsample(×2) + Conv2d(128→64, 3×3)        → 32×32
  ↓ BatchNorm2d + ReLU
  ↓ Upsample(×2) + Conv2d(64→3, 3×3)          → 64×64
  ↓ Tanh  →  output ∈ [−1, 1]³
```

The `Tanh` output activation ensures generated pixel values stay in `[−1, 1]`, matching the normalised real image range.

### 3.5 Baseline Discriminator Architecture

The discriminator is a 4-stage strided convolutional network that downsamples `64×64 → 1×1`:

```
Input (3×64×64)
  ↓ Conv2d(3→64, 4×4, stride=2)    → 32×32
  ↓ LeakyReLU(0.2)
  ↓ Conv2d(64→128, 4×4, stride=2)  → 16×16
  ↓ BatchNorm2d + LeakyReLU(0.2)
  ↓ Conv2d(128→256, 4×4, stride=2) → 8×8
  ↓ BatchNorm2d + LeakyReLU(0.2)
  ↓ Conv2d(256→512, 4×4, stride=2) → 4×4
  ↓ BatchNorm2d + LeakyReLU(0.2)
  ↓ Conv2d(512→1, 4×4, stride=1)   → 1×1 (logit)
  ↓ .view(-1)  →  scalar logit per image
```

LeakyReLU with slope 0.2 is used in the discriminator because it allows gradient flow even for negative activations, preventing dead neurons — a recommendation from the original DCGAN paper [^1].

### 3.6 Weight Initialisation

All convolutional weights are initialised from 𝒩(0, 0.02) and all BatchNorm weights from 𝒩(1, 0.02) with bias set to 0. This is the exact scheme specified in Radford et al. (2015) [^1], chosen empirically to produce stable early-training dynamics.

### 3.7 Loss Function

`nn.BCEWithLogitsLoss` (binary cross-entropy with logits) is used throughout. The discriminator outputs raw logits (no sigmoid), and the loss function combines sigmoid and BCE in a numerically stable fused form. This is preferable to applying sigmoid externally followed by `BCELoss`, as it avoids log-of-near-zero values (Paszke et al., 2019) [^8].

### 3.8 Optimiser

Adam with β₁ = 0.5 and β₂ = 0.999. The β₁ = 0.5 recommendation comes directly from the DCGAN paper [^1] and differs from the default of 0.9 used in most non-GAN settings. Lower β₁ reduces the momentum of the first moment estimate, which helps prevent oscillations in the adversarial game.

### 3.9 Evaluation Artefacts

Every run saves:
- A **fixed noise vector** (64 latent codes, same across all epochs) used to track generator progress on identical inputs.
- **PNG sample grids** every 25 epochs.
- **Periodic checkpoints** every 5–10 epochs.
- **FID/KID** computed every 50 epochs and at the final epoch.
- A **latent interpolation** (linear interpolation between two random z vectors) saved at the end of training to diagnose latent space structure.

---

## 4. g0 — Baseline DCGAN (Instrumented)

### 4.1 Configuration

| Parameter | Value |
|---|---|
| `NUM_EPOCHS` | 200 |
| `LR_G` | 2×10⁻⁴ |
| `LR_D` | 1×10⁻⁴ |
| `START_FOLD / END_FOLD` | 0 / 5 |
| EMA | ✗ |
| Label smoothing | ✗ |
| SpectralNorm | ✗ |
| Dropout in D | ✗ |

### 4.2 Purpose

g0 is the **measurable baseline**. No improvements are introduced relative to a standard DCGAN. Its sole purpose is to establish reproducible FID/KID numbers, loss curves, and visual samples against which all subsequent iterations can be compared. Without a rigorous baseline, it is impossible to know whether any subsequent change represents genuine improvement or noise.

### 4.3 Learning Rates: Asymmetric Schedule (LR_G > LR_D)

A deliberate asymmetry was applied from the start: `LR_G = 2e-4` while `LR_D = 1e-4`. This is consistent with the **Two Time-Scale Update Rule (TTUR)** introduced by Heusel et al. (2017) [^2]. Under TTUR, using a higher learning rate for G than D can slow D's convergence slightly relative to G, preventing the discriminator from becoming perfectly accurate before the generator has had a chance to learn useful gradients. Heusel et al. showed theoretically that TTUR promotes convergence to a local Nash equilibrium.

### 4.4 Dataset: Folds 0–5

The DFF dataset is structured into folds. At baseline, only **folds 0–5** were used, representing a small subset of the available 30,000 real images.

This conservative starting point was intentional: it allows g1 to test the hypothesis that **data scale** is a limiting factor. If g0's poor results can be substantially explained by insufficient training data, the improvement from adding more data will be isolated and measurable.

### 4.5 Results

| Epoch | FID | Notes |
|---|---|---|
| 50 | ~328.8 | Very high, noisy images |
| 150 | ~300.4 | Best FID observed |
| 200 | ~307.2 | FID **increased** from epoch 150 |

```
D(x)         ≈ 0.98   (discriminator near-certain about real images)
D(G(z))_fake ≈ 0.01   (discriminator near-certain about fakes)
D(G(z))_gen  ≈ 0.01
Loss_D       ≈ 0       (near-zero discriminator loss)
Loss_G       >> 0      (generator receives almost no useful gradient)
```

### 4.6 Interpretation

The discriminator **dominated completely** by epoch ~50. The training dynamic collapsed into a regime where D(x) → 1 and D(G(z)) → 0, meaning the generator's loss gradient was saturated and nearly uninformative. This is the classic **vanishing gradient problem** in GAN training, described in the original GAN paper by Goodfellow et al. (2014) [^4].

Visually, the epoch-200 samples showed rough portrait structure (skin tones, approximate head silhouettes, some colour consistency) but no coherent facial features. The latent interpolation showed smooth transitions in colour and texture but no semantic facial attributes, indicating the latent space had not learned a structured face manifold.

The FID **worsening between epochs 150 and 200** is noteworthy: the generator's quality degraded in the final 50 epochs, consistent with training instability once the discriminator has fully saturated.

### 4.7 Decision

Two root causes were identified:
1. **Insufficient training data**: only folds 0–5, which likely caused overfitting of D to a small real distribution.
2. **Adversarial imbalance**: D too strong relative to G.

The first cause is cheaper to test (no architecture changes required) and more fundamental. Therefore, g1 addresses **data scale** first, keeping everything else constant.

---

## 5. g1 — More Data (Folds 0–20)

### 5.1 Configuration Changes from g0

| Parameter | g0 | g1 | Change |
|---|---|---|---|
| `START_FOLD / END_FOLD` | 0 / 5 | **0 / 20** | +15 folds of real images |

All other hyperparameters, architecture, and training procedures are identical to g0.

### 5.2 Motivation

The central hypothesis tested in g1:

> **If the main limitation of g0 is insufficient real data, then increasing the number of real training images should meaningfully lower FID without any other changes.**

This hypothesis is grounded in a fundamental principle of deep learning: models trained on more diverse data generalise better and learn richer representations. In the GAN setting, a discriminator trained on more diverse real images provides a richer and less easily-saturated gradient signal to the generator.

Specifically:
- **Discriminator generalisation**: With only folds 0–5, the discriminator may memorise a limited set of real images and saturate quickly. More real images force D to learn a more distributed, general notion of "real", producing more informative gradients for G.
- **Generator diversity**: G implicitly tries to match the distribution of real images. A broader real distribution encourages G to produce more varied outputs, reducing mode collapse risk.

This is consistent with the observations of Brock et al. (2019) [^9] in BigGAN, where a key finding was that **dataset scale is one of the most impactful factors in GAN image quality**.

### 5.3 Results

| Epoch | FID | KID mean | Comparison to g0 |
|---|---|---|---|
| 50 | 260.38 | 0.1389 | −68 vs g0@50 |
| 100 | 219.45 | 0.0980 | — |
| 150 | 217.53 | 0.0908 | — |
| 200 | **208.99** | **0.0887** | −91 vs g0 best FID |

```
D(x)         ≈ 0.991  (still near-perfect)
D(G(z))_fake ≈ 0.009
D(G(z))_gen  ≈ 0.005
Loss_G final ≈ 6.14
Loss_D final ≈ 0.018
```

### 5.4 Interpretation

The hypothesis was confirmed decisively. FID improved by **~91 points** relative to the best g0 value, purely by adding more real training images. The generator now produces images with clearer human-like structure: body silhouettes, approximate head/face regions, portrait-like colour contrast, and clothing/background separation.

However, the discriminator **remains completely saturated**. The adversarial dynamic is essentially unchanged from g0 — D still classifies all real and fake images with near-certainty. This tells us that while more data improved the quality ceiling, the **training dynamics problem** was not resolved.

The FID curve also continues to improve through epoch 200 (unlike g0, which peaked at epoch 150 and degraded), suggesting the model had not yet reached its limit with this data volume.

### 5.5 Decision

g1 establishes that data scale is the most impactful single variable. This configuration (folds 0–20) is now the baseline for g2, g3, and g4. The remaining problem — discriminator saturation — will be addressed next.

---

## 6. g2 — Discriminator Balancing (Negative Result)

### 6.1 Configuration Changes from g1

| Parameter | g1 | g2 | Change |
|---|---|---|---|
| `LR_D` | 1×10⁻⁴ | **5×10⁻⁵** | Halved D learning rate |
| `REAL_LABEL` | 1.0 | **0.9** | One-sided label smoothing |
| `FAKE_LABEL` | 0.0 | 0.0 | Unchanged |

### 6.2 Motivation

The problem observed at the end of g1: discriminator saturation. Two targeted interventions were applied simultaneously to reduce D's dominance:

#### 6.2.1 Reduced Discriminator Learning Rate

Slowing D's learning rate gives G more time to catch up before D reaches saturation. This is consistent with Heusel et al.'s Two Time-Scale Update Rule [^2]: widening the ratio between LR_G and LR_D biases the training dynamics toward generator-favourable convergence. The specific choice of halving LR_D from 1e-4 to 5e-5 was conservative enough to avoid making D useless, but significant enough to visibly reduce saturation speed.

#### 6.2.2 One-Sided Label Smoothing on Real Targets

Instead of training D with hard targets of 1.0 for real images, the real target was set to **0.9**. Fake targets remain 0.0 (asymmetric/one-sided smoothing).

This technique was introduced by Salimans et al. (2016) in "Improved Techniques for Training GANs" [^10], one of the most cited papers on GAN stabilisation. Their rationale: hard labels of exactly 1.0 push D's logit output toward +∞ for real images, which causes excessively confident gradients. Smoothing the real target to 0.9 limits this effect without reducing D's ability to distinguish fakes (the 0.0 target for fakes is left unchanged, hence "one-sided").

The paper also notes that two-sided smoothing (smoothing both real and fake targets) can be harmful because smoothed fake targets inadvertently reward G for producing images that look *somewhat* like fakes — which is the opposite of what G should optimise for.

### 6.3 Results

| Epoch | FID | KID mean | Comparison to g1 |
|---|---|---|---|
| 50 | 264.99 | 0.1502 | Worse |
| 100 | 249.94 | 0.1300 | Worse |
| 150 | 249.98 | 0.1273 | Worse |
| 200 | 244.18 | 0.1317 | **+35 FID vs g1 (worse)** |

```
D(x)         ≈ 0.85–0.90  (less saturated, as intended)
D(G(z))_fake ≈ ~0.10
Loss_D       ≈ higher than g1 (D no longer trivially succeeds)
Loss_G       ≈ lower than g1 (less extreme)
```

### 6.4 Interpretation

The intervention produced **exactly the intended effect on D's saturation**, but this did not translate into better FID. The discriminator became less dominant — D(x) dropped from ~0.99 to ~0.85-0.90, and D(G(z)) was no longer near-zero. However, FID worsened significantly compared to g1.

This reveals an important insight: **the discriminator becoming too weak is equally harmful as it becoming too strong**. When D is over-regularised, it no longer provides a sufficiently informative loss landscape for G to navigate. The gradient signal from D must be non-trivial (D should not trivially accept all images as real) but also non-zero (D should not trivially reject all fakes with certainty).

g2 is a documented **negative result**, but a valuable one: it rules out the naive approach of simply slowing D down. The next step requires a smarter regularisation strategy — one that constrains D's capacity without simply reducing its learning speed.

### 6.5 Decision

Discard g2 configuration. Return to g1 (folds 0–20, LR_D = 1e-4, no label smoothing) as the base for the next improvement.

---

## 7. g3 — EMA on the Generator

### 7.1 Configuration Changes from g1

| Parameter | g1 | g3 | Change |
|---|---|---|---|
| `USE_EMA` | ✗ | **✓** | EMA maintained on generator weights |
| `EMA_DECAY` | — | **0.999** | Exponential moving average decay |

Training dynamics, architecture, data, and all other parameters remain identical to g1.

### 7.2 Motivation

Even when a GAN trains stably at the macro level, the generator's weights undergo high-frequency oscillations at the mini-batch level. These oscillations mean that the **instantaneous weights at the end of training may not correspond to the best generative quality seen during training**. The generator might have briefly reached a higher-quality state mid-training that was subsequently disrupted by the adversarial dynamics.

Exponential Moving Average (EMA) of model parameters is a post-processing technique that maintains a shadow copy of the generator weights as a weighted average over all past update steps:

```
θ_ema ← decay × θ_ema + (1 − decay) × θ_current
```

With `decay = 0.999`, the EMA weights represent approximately the last 1/(1−0.999) = 1000 update steps worth of history, heavily weighted toward recent steps. This smooths out high-frequency noise in the weight trajectory.

**Literature support**: Karras et al. (2019) in StyleGAN [^11] showed that evaluating the EMA generator instead of the raw generator consistently produces better FID across all training durations. The EMA generator was used for all sample generation and metric computation in StyleGAN and its successors. The EMA approach is also used in BigGAN (Brock et al., 2019) [^9].

**Key property**: EMA does not change the training procedure at all — D still trains against the raw generator G, not the EMA copy. EMA is applied **only for evaluation and sample generation**. This means EMA cannot introduce instability into the adversarial game; it can only improve the quality of the model used for inference.

### 7.3 Implementation

After every `optimizerG.step()`:

```python
for ema_param, param in zip(netG_ema.parameters(), netG.parameters()):
    ema_param.data.mul_(decay).add_(param.data, alpha=1.0 - decay)
```

BatchNorm running statistics (running_mean, running_var) are copied directly from the live generator to the EMA generator, rather than being maintained separately. This is important because running statistics track the actual data distribution and should not be averaged with historical values.

### 7.4 Results

| Epoch | FID (EMA evaluated) | KID mean | Comparison to g1 |
|---|---|---|---|
| 50 | 267.57 | 0.1660 | — |
| 100 | 223.46 | 0.1048 | — |
| 150 | 221.13 | 0.0962 | — |
| 200 | **205.77** | **0.0889** | **−3.2 FID vs g1** |

```
D(x)         ≈ 0.9916   (D still dominates — EMA does not change adversarial dynamics)
D(G(z))_fake ≈ 0.0083
D(G(z))_gen  ≈ 0.0049
```

### 7.5 Interpretation

EMA produced a **small but consistent and reproducible improvement** in FID (205.77 vs 208.99). The FID curve is also smoother: unlike g0 which degraded after epoch 150, g3's FID continues improving monotonically through epoch 200. KID is approximately unchanged.

Visually, the EMA generator produces slightly cleaner samples than the raw generator at the same epoch: fewer extreme artefacts and more consistent texture. The latent interpolation also shows slightly smoother transitions.

Critically, EMA did not change D's behaviour at all, confirming that it operates purely at the evaluation level. The discriminator remains saturated, meaning the underlying adversarial imbalance is still present. EMA is a **free quality improvement** that costs only memory (one extra generator copy) and should be kept in all future iterations.

### 7.6 Decision

Retain EMA with decay=0.999. g3 is the new best configuration. The discriminator saturation problem remains unsolved — next, we try a principled architectural regularisation of D.

---

## 8. g4 — Spectral Normalization on Discriminator (Negative Result)

### 8.1 Configuration Changes from g3

| Parameter | g3 | g4 | Change |
|---|---|---|---|
| Discriminator conv layers | Plain `Conv2d` | **`spectral_norm(Conv2d)`** | Lipschitz constraint via SN |
| Discriminator BatchNorm | Present in 3 layers | **Removed entirely** | SN replaces BN's role in D |
| `USE_EMA` | ✓ | ✓ | Retained |

### 8.2 Motivation

The core problem — discriminator saturation — was identified consistently across g0–g3. A principled approach to prevent D from becoming arbitrarily confident is to constrain its **Lipschitz constant**.

Spectral Normalization (SN), introduced by Miyato et al. (2018) [^12] in "Spectral Normalization for Generative Adversarial Networks", constrains the spectral norm (largest singular value) of each weight matrix to be at most 1. This enforces a 1-Lipschitz constraint on the discriminator, meaning the discriminator cannot change its output by more than the change in input — effectively capping how confident D can be about any single input.

Formally, for a weight matrix **W**, spectral normalisation replaces **W** with **W/σ(W)** where σ(W) is the largest singular value, estimated efficiently via power iteration. This is applied per-layer.

**Advantages of SN over reducing LR_D** (as attempted in g2):
- SN constrains the functional space of D rather than just slowing its dynamics. It prevents D from memorising individual training examples by constraining how sharply D's decision boundary can change.
- SN is compatible with the original learning rate, so the adversarial game's time-scale relationship is preserved.
- Miyato et al. demonstrated that SN-regularised discriminators produce more stable training and better FID than BatchNorm-only discriminators on several benchmarks.

**Why BatchNorm was removed from D**: BatchNorm in the discriminator is known to cause instability because it introduces correlations between samples within a mini-batch, making D's output for one sample depend on other samples in the batch. Radford et al. [^1] originally excluded BN from the first discriminator layer for this reason. SN provides an alternative normalisation mechanism that does not have this side-effect.

### 8.3 Implementation

```python
def conv(in_channels, out_channels, kernel_size, stride, padding):
    layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
    return spectral_norm(layer)
```

All five convolutional layers in D are wrapped with `spectral_norm`. BatchNorm layers are removed entirely from the discriminator.

### 8.4 Results

| Epoch | FID (EMA) | KID mean | Comparison to g3 |
|---|---|---|---|
| 50 | 295.04 | 0.2014 | Much worse early |
| 100 | 288.58 | 0.1724 | — |
| 150 | 254.00 | 0.1367 | — |
| 200 | **228.51** | **0.1124** | **+22.7 FID vs g3 (worse)** |

```
D(x)         ≈ 0.65    (dramatically less saturated — SN worked)
D(G(z))_fake ≈ 0.51
D(G(z))_gen  ≈ 0.49
```

### 8.5 Interpretation

SN achieved its intended goal: D(x) dropped from ~0.99 (g3) to ~0.65, and D(G(z)) rose from ~0.005 to ~0.49. The adversarial game is now much closer to the theoretical Nash equilibrium of D(x) = D(G(z)) = 0.5. In this sense, g4 was architecturally successful.

However, FID **worsened significantly**: 228.51 vs 205.77 in g3. The generated images became visually "washed out" — less defined, softer, and less structured than g3's outputs.

The explanation is that SN constrained D so strongly that it could no longer provide the high-gradient signal G needs to learn fine-grained image structure. The 1-Lipschitz constraint is a global bound: while it prevents saturation, it also reduces D's discrimination capability for subtle features. With a near-uniform D that assigns ~0.5 probability to all images, G has no clear direction to improve fine image quality.

This is the same fundamental tension observed in g2, approached from a different angle. The lesson: **discriminator regularisation must be calibrated**, not maximised. A perfectly balanced GAN (D outputs exactly 0.5 for everything) conveys no useful information to G.

### 8.6 Decision

g4 is documented as a negative/partial result. g3 (EMA only, no SN in D) remains the best configuration. The SpectralNorm on D approach is abandoned in isolation. However, the concept of SN is revisited later in g8 combined with other techniques.

---

## 9. g5 — Full Dataset + Best-FID Checkpoint

### 9.1 Configuration Changes from g3

| Parameter | g3 | g5 | Change |
|---|---|---|---|
| `START_FOLD / END_FOLD` | 0 / 20 | **0 / 99** | All available folds |
| Best checkpoint selection | ✗ | **✓** | Saved whenever new best FID achieved |
| `USE_EMA` | ✓ | ✓ | Retained |

### 9.2 Motivation

From the g1 experiment, we know that **data scale was the single largest driver of FID improvement** (−91 FID points vs g0). g1 used folds 0–20; g5 extends this to all available folds (0–99), which covers the full 30,000 real images in the dataset.

#### 9.2.1 Why More Data Continues to Help

In g1 → g5 (folds 0–20 → 0–99), the number of unique real images increases substantially. The benefits are:
1. **Richer real distribution**: The discriminator sees a more diverse set of real faces, forcing it to learn general facial features rather than memorising a subset.
2. **Reduced generator overfitting**: G has more variation to match, encouraging it to model a broader face distribution rather than converging to a small set of modes.
3. **Slower D saturation**: With more diverse real images, D takes longer to reach near-perfect accuracy, providing a longer window of useful gradient signal.

This is consistent with the general deep learning finding — articulated explicitly in BigGAN [^9] and in image generation surveys (Borji, 2022) [^13] — that large, diverse datasets are fundamental to high-quality generation.

#### 9.2.2 Best-FID Checkpoint Selection

In g0 and g1, the **final epoch checkpoint** was used as the deployed model. However, the g0 results showed that FID can **increase** at later epochs (epoch 150 → epoch 200), meaning the best generator is not necessarily the one at the end of training.

Saving the checkpoint with the **lowest observed FID** throughout training ensures the deployed model is the best-seen state, not simply the last. This is standard practice in generative modelling (Karras et al., 2019) [^11] and removes the dependency on choosing the right number of epochs in advance.

### 9.3 Results

| Epoch | FID (EMA) | KID mean | Notes |
|---|---|---|---|
| 50 | — | — | — |
| 100 | **~163.31** | — | Best FID checkpoint saved |
| 150 | — | — | — |
| 200 | ~170.49 | ~0.0483 | FID worse than epoch 100 |

The **best checkpoint was epoch 100**, not epoch 200 — validating the need for best-FID checkpoint selection.

### 9.4 Comparison to All Previous Iterations

| Iteration | Best FID | Data Folds |
|---|---|---|
| g0 | ~300.4 | 0–5 |
| g1 | 208.99 | 0–20 |
| g3 (EMA) | 205.77 | 0–20 |
| **g5** | **~163.31** | **0–99** |

The improvement from g3 to g5 (~42 FID points) is the second largest single-step improvement in the project, after the g0→g1 jump. This confirms that **data scale remains the dominant factor even at larger volumes**.

### 9.5 Interpretation

Visually, g5 samples show more human-like structure: clearer head shapes, skin tone regions, some approximate facial areas, and more consistent portrait composition. However, the images remain blurry and incoherent at the fine-feature level. Discriminator saturation is still visible:

```
D(x)         ≈ 0.98
D(G(z))_gen  ≈ 0.008
Loss_G       ≈ 7.21   (very high — G is still being strongly rejected)
Loss_D       ≈ 0.06
```

The generator is learning, but still fighting against an overwhelmingly dominant discriminator. The best-FID checkpoint falling at epoch 100 (rather than 200) suggests that D saturates early and the generator's improvement stalls or reverses in the later training phase.

### 9.6 Decision

g5 is the new best model. The persistent adversarial imbalance and the FID degradation after epoch 100 motivate a bundle of discriminator regularisation techniques in g6 — informed by what failed individually in g2 and g4.

---

## 10. g6 — Face Crop + Discriminator Regularisation Bundle

### 10.1 Configuration Changes from g5

| Parameter | g5 | g6 | Change |
|---|---|---|---|
| `START_FOLD / END_FOLD` | 0 / 99 | **0 / 99** | Unchanged |
| Transform | Standard centre crop | **Face crop (Haar) + centre crop fallback** | Domain-specific preprocessing |
| `REAL_LABEL` | 1.0 (hard) | **Uniform [0.85, 1.0]** | One-sided stochastic label smoothing |
| Instance noise | ✗ | **σ=0.05 decaying over 80 epochs** | Input perturbation |
| Dropout in D | ✗ | **Dropout2d(p=0.20)** in first 3 D layers | Structural regularisation |
| `USE_EMA` | ✓ | ✓ | Retained |
| `NUM_EPOCHS` | 200 | 200 | Unchanged |

### 10.2 Motivation

The g5 diagnosis identified two distinct problems requiring different interventions:

1. **Distribution complexity**: The DFF IMDB-WIKI dataset contains images of highly variable framing — full bodies, crowd scenes, sports photos, group photos, and close-up faces all in the same training set. A 64×64 DCGAN has limited representational capacity; training on such a wide distribution forces the generator to spread probability mass thinly across many different image types, resulting in blurry, incoherent outputs.

2. **Discriminator saturation**: D dominates from early epochs, suppressing G's useful gradient signal.

g6 attacks both problems with a bundled approach, applying multiple techniques whose effects are expected to compound.

#### 10.2.1 Face Crop Preprocessing

Using OpenCV's Haar Cascade face detector (`haarcascade_frontalface_default.xml`), images are **cropped to the detected face region** with a 45% margin before resizing to 64×64. When no face is detected, a **square centre crop** is used as fallback.

**Justification**: By cropping to faces, the training distribution is simplified from "diverse celebrity images of any framing" to "approximately face-centred images". This narrows the target distribution that G must learn to approximate, which is especially important for a low-capacity 64×64 DCGAN.

This is consistent with the design of face-specific GAN datasets such as CelebA (Liu et al., 2015) [^14], which provides pre-aligned, cropped face images specifically to reduce the distribution complexity that generative models must handle. The aligned CelebA format is one of the standard DCGAN benchmarks.

Importantly, face detection may fail for many images (sports scenes, non-frontal faces, group shots), and the centre-crop fallback ensures training continues with the full dataset regardless.

#### 10.2.2 One-Sided Stochastic Label Smoothing

Instead of deterministic label smoothing (REAL_LABEL = 0.9 as in g2), g6 uses **stochastic** smoothing: the real target is sampled uniformly from [0.85, 1.0] per batch:

```python
real_targets = torch.empty(batch_size, device=DEVICE).uniform_(0.85, 1.0)
```

**Justification**: Salimans et al. (2016) [^10] recommend label smoothing to prevent D from becoming overconfident on real images. The stochastic variant was chosen over the deterministic g2 variant because it introduces additional regularisation noise, preventing D from precisely calibrating its output to a fixed smoothed target. The range [0.85, 1.0] was chosen to keep the real signal strong (D should still strongly prefer real images) while preventing extreme logit values. Fake targets remain hard at 0.0 (one-sided, as recommended in [^10]).

#### 10.2.3 Instance Noise with Linear Decay

Gaussian noise is added to both real and generated images before they are passed to D:

```
x_noisy = x + ε,   ε ~ N(0, σ²)
σ = σ₀ × (1 − epoch/decay_epochs),   σ₀ = 0.05,   decay_epochs = 80
```

σ starts at 0.05 at epoch 1 and linearly decays to 0 at epoch 80, then remains 0.

**Justification**: Adding noise to D's inputs is theoretically motivated by Sønderby et al. (2016) [^15] (and independently by Roth et al., 2017 [^16]). The key insight is that when D operates on noisy inputs, the effective "distance" between the real and generated distributions is reduced (both are convolved with the same noise distribution). This prevents D from trivially distinguishing real and fake images based on subtle low-level statistics early in training, giving G time to learn useful image structure before D saturates. The linear decay ensures that as G matures and generates more realistic images, D is eventually evaluated on clean inputs, providing the full discriminative signal needed for later-stage refinement.

#### 10.2.4 Dropout in the Discriminator

Dropout2d (channel-wise dropout) with p=0.20 is applied after the LeakyReLU in the first three convolutional blocks of D.

**Justification**: Dropout as a regulariser in the GAN discriminator has been explored in several works. The intuition follows directly from standard deep learning: dropout prevents any single activation from becoming the dominant decision signal, forcing D to use distributed representations. For GANs specifically, this prevents D from memorising local discriminative features of the (limited) training images and instead forces it to learn more general structural features. A dropout rate of 0.20 was chosen conservatively — high enough to provide regularisation, low enough not to excessively degrade D's ability to distinguish real from fake.

### 10.3 Results

| Epoch | FID (EMA) | KID mean | Notes |
|---|---|---|---|
| 1 | 368.77 | — | Initial |
| 50 | 181.60 | — | |
| 100 | 162.95 | — | |
| 150 | 156.16 | — | |
| 200 | **148.20** | **0.0435** | Best FID — final epoch |

```
Loss_G final  ≈ 4.40   (vs 7.21 in g5 — significantly healthier)
Loss_D final  ≈ 0.41   (vs 0.06 in g5 — D no longer trivially wins)
D(x)          ≈ 0.88   (vs 0.98 — less saturated)
D(G(z))_gen   ≈ 0.037  (vs 0.008 — G getting more signal)
```

**Crucially, the best checkpoint is at epoch 200 — the final epoch**. Unlike g5 (where best FID was at epoch 100), g6's FID continues improving throughout the full training duration, indicating that discriminator regularisation successfully extended the window of productive adversarial training.

### 10.4 Comparison

| Iteration | Best FID | Notes |
|---|---|---|
| g5 | ~163.31 @ epoch 100 | FID degrades after epoch 100 |
| **g6** | **148.20 @ epoch 200** | FID improves monotonically; +9.3% improvement |

### 10.5 Interpretation

The bundle of face crop + label smoothing + instance noise + dropout produced a **9.3% FID improvement** over g5. The most meaningful indicator of improvement is not just the FID number but the **training dynamics**: in g6, Loss_G dropped from ~7.2 to ~4.4 and Loss_D rose from ~0.06 to ~0.41. This means the adversarial game became substantially more balanced — the generator received a less saturated gradient signal throughout training.

The face detection quality was imperfect (many images without clear faces received centre crops, and some misclassified crops included full bodies or group scenes), meaning the distribution simplification effect was only partial. Nevertheless, even partial simplification and the regularisation package produced consistent improvement.

### 10.6 Decision

g6 is the new best configuration. FID is still improving at epoch 200, suggesting the model could benefit from more training epochs. The next iteration tests extending the training budget.

---

## 11. g7 — Extended Training (400 Epochs, Regularised)

### 11.1 Configuration Changes from g6

| Parameter | g6 | g7 | Change |
|---|---|---|---|
| `NUM_EPOCHS` | 200 | **400** | Double training duration |
| `START_FOLD / END_FOLD` | 0 / 99 | **0 / 5** | Reduced to smaller fold set |
| All regularisation (noise, smoothing, dropout, EMA) | ✓ | ✓ | Retained from g6 |

### 11.2 Motivation

The g6 results demonstrated that FID was still improving at the end of training (epoch 200 was the best checkpoint). This is a strong signal that the model had not yet converged: **training budget was a limiting factor**.

#### 11.2.1 Why More Epochs Can Help

In GANs with regularised discriminators, convergence is slower than in unregularised settings because each update step provides a smaller gradient magnitude (D is prevented from reaching extreme confidence). This means the generator requires more steps to traverse the loss landscape toward high-quality generation.

Doubling from 200 to 400 epochs tests whether the g6 training trajectory — which was still improving at epoch 200 — continues to improve with more time. This is motivated by the observation in StyleGAN [^11] and other large-scale GAN works that substantial improvements in FID continue to accumulate over hundreds of thousands of training steps.

#### 11.2.2 Reduced Fold Count

The fold count was reduced back to 0–5 for g7. This was a practical decision: with 400 epochs, each epoch over folds 0–99 takes significantly longer. Folds 0–5 provide a faster per-epoch iteration cycle, allowing more epochs in the same wall-clock time. This is a **compute/data trade-off**: more training steps vs. more data diversity per step.

### 11.3 Context

g7 evaluates whether the g6 dynamics — which showed monotonically improving FID — continue to improve given double the training time. The regularisation stack from g6 is fully preserved.

---

## 12. g8 — Improved Discriminator (SpectralNorm + MinibatchStd)

### 12.1 Configuration Changes from g6/g7

| Parameter | g6 | g8 | Change |
|---|---|---|---|
| `NUM_EPOCHS` | 200 | **400** | Extended training |
| `START_FOLD / END_FOLD` | 0 / 99 | **0 / 5** | Reduced to manage runtime |
| Discriminator architecture | BatchNorm + Dropout | **SpectralNorm + MinibatchStd + Dropout** | Combined architectural upgrade |
| BatchNorm in D | ✓ | **✗** | Removed (SN replaces) |
| SpectralNorm on D convs | ✗ | **✓** | All D conv layers |
| MinibatchStd layer | ✗ | **✓** | Inserted before final classifier |

### 12.2 Motivation

g4 showed that applying Spectral Normalization alone to D degraded FID because D became too weakly constrained to guide G. However, g6 showed that discriminator regularisation is beneficial when it is **combined with other stabilising techniques** (instance noise, label smoothing, dropout). g8 revisits SpectralNorm in this richer context, combining it with the full g6 regularisation stack and adding a new technique: **MinibatchStd**.

#### 12.2.1 Spectral Normalization (Revisited in Context)

As established in the g4 section, SN constrains D's Lipschitz constant by normalising each weight matrix by its spectral norm. The reason for revisiting SN here, despite its failure in g4, is that in g4 it was the **only** regularisation applied to D. In g8, it coexists with instance noise, label smoothing, and dropout — all of which also prevent D saturation. The combined effect should produce a more calibrated D that is constrained but not weakened.

Furthermore, Miyato et al. (2018) [^12] demonstrate SN's benefits on discriminators without BatchNorm. Since g8 also removes BatchNorm from D (as recommended in several works on GAN stabilisation), SN and the removal of BN act synergistically.

#### 12.2.2 MinibatchStd Layer

Introduced by Karras et al. (2018) in "Progressive Growing of GANs" [^5], MinibatchStd addresses **mode collapse** — the failure mode where G learns to produce only a small subset of the possible image space.

The layer computes a scalar statistic (mean standard deviation) across the mini-batch for each spatial location and channel, then concatenates this statistic as an extra channel to the feature map before the final classification layer:

```python
class MinibatchStd(nn.Module):
    def forward(self, x):
        std = x.std(dim=0, keepdim=True)           # (1, C, H, W)
        std = std.mean(dim=1, keepdim=True)         # (1, 1, H, W)
        std = std.expand(x.size(0), 1, x.size(2), x.size(3))
        return torch.cat([x, std], dim=1)           # (B, C+1, H, W)
```

**How it fights mode collapse**: If G is in mode collapse (generating nearly identical images), the per-batch std will be very low. D can detect this low diversity signal and assign lower probability to the batch, creating a gradient that pushes G to increase diversity. Crucially, MinibatchStd gives D a **diversity signal without making it stronger at individual-sample classification** — it does not increase D's ability to reject any single sample, only batches that are insufficiently diverse.

This is the same mechanism used in the standard minibatch discrimination technique introduced by Salimans et al. (2016) [^10], but in a computationally simpler form. Karras et al. adopted it in Progressive GAN [^5] and retained it in StyleGAN [^11] because of its effectiveness at maintaining generator diversity.

#### 12.2.3 Architecture After Changes

```
Input (3×64×64)
  ↓ SN(Conv2d(3→64, 4×4, stride=2))   → 32×32
  ↓ LeakyReLU(0.2) + Dropout2d(0.3)
  ↓ SN(Conv2d(64→128, 4×4, stride=2)) → 16×16
  ↓ LeakyReLU(0.2) + Dropout2d(0.3)
  ↓ SN(Conv2d(128→256, 4×4, stride=2))→ 8×8
  ↓ LeakyReLU(0.2) + Dropout2d(0.3)
  ↓ SN(Conv2d(256→512, 4×4, stride=2))→ 4×4
  ↓ LeakyReLU(0.2)
  ↓ MinibatchStd()                     → 4×4, 513 channels
  ↓ SN(Conv2d(513→1, 4×4, stride=1))  → 1×1
  ↓ .view(-1)
```

Note: the dropout rate is 0.30 in g8 (vs 0.20 in g6), adjusted because SN already constrains each layer's Lipschitz constant and a slightly higher dropout rate better compensates.

### 12.3 Sample Output Format

g8 also introduces an engineering improvement to the sample recording: rather than saving a single large JSONL file, samples are split into ~90 MB chunks using a calibrated `EPOCHS_PER_SPLIT` calculation based on the measured per-frame size. This prevents file size issues during long (400-epoch) training runs.

### 12.4 Expected Outcomes

- SN should prevent D from saturating, as in g4 — but now within the g6 regularisation context that prevents D from becoming too weak.
- MinibatchStd should maintain generator diversity throughout 400 epochs of training, reducing the risk of mode collapse in later epochs.
- The combination with instance noise and label smoothing from g6 should keep the adversarial game calibrated throughout the extended training.

---

## 13. Global Summary Table

### 13.1 Chronological FID Progression

| Iteration | Key Change | Best FID | Best Epoch | FID @ ep.200 | Decision |
|---|---|---|---|---|---|
| g0 | DCGAN baseline | ~300.4 | 150 | ~307.2 | Baseline; D dominates |
| g1 | Folds 0–5 → 0–20 | 208.99 | 200 | 208.99 | Large improvement; keep |
| g2 | LR_D halved + label smooth 0.9 | 244.18 | 200 | 244.18 | **Negative result**: D too weak |
| g3 | EMA (decay=0.999) on G | 205.77 | 200 | 205.77 | Small improvement; keep EMA |
| g4 | SpectralNorm on D, remove BN | 228.51 | 200 | 228.51 | **Negative result**: D too constrained alone |
| g5 | Folds 0–20 → 0–99 + best-FID ckpt | ~163.31 | 100 | ~170.49 | Large improvement; keep |
| g6 | Face crop + noise + smoothing + dropout | **148.20** | 200 | 148.20 | Best; FID monotonically improves |
| g7 | 400 epochs + regularisation (folds 0–5) | — | — | — | Extended training test |
| g8 | SN + MinibatchStd + 400 epochs + reg. | — | — | — | Architectural upgrade |

### 13.2 Key Lessons from Each Dimension

#### Image Size (64×64 throughout)
Fixed at 64×64 for all iterations. This resolution is the natural upper bound for the DCGAN architecture without architectural extensions (progressive growing, attention). Increasing it would require additional upsampling stages in G and additional downsampling in D, and would not improve FID without a correspondingly larger architecture and dataset.

#### Epochs (200 → 400 in g7/g8)
200 epochs was insufficient given the g6 regularisation: FID was still improving at epoch 200. The extension to 400 epochs in g7/g8 is motivated by the observation that regularised adversarial training converges more slowly, requiring more steps to reach the same quality level.

#### Sample Size / Dataset Scale
The most impactful dimension in the entire project:
- g0 → g1: folds 0–5 → 0–20, −91 FID points
- g3 → g5: folds 0–20 → 0–99, −42 FID points
- More data consistently and substantially improves both FID and KID.

#### Architectural Changes
- **Generator**: Unchanged throughout (Upsample+Conv, 4 stages to 64×64, Tanh output, BatchNorm throughout).
- **Discriminator g0–g5**: Standard strided-conv + BatchNorm + LeakyReLU.
- **Discriminator g6**: Added Dropout2d(0.20) to first 3 layers.
- **Discriminator g8**: Full replacement with SN + MinibatchStd + Dropout(0.30), no BatchNorm.

#### Optimisations
- **EMA (g3+)**: Free quality improvement with no training cost; consistently kept.
- **Best-FID checkpoint (g5+)**: Ensures deployed model is best-seen, not last-seen.
- **JSONL split sampling (g8)**: Engineering improvement to handle 400-epoch recording.

#### Stability Changes
- **g2**: LR_D reduction + deterministic label smooth → D became too weak (negative result).
- **g6**: Stochastic label smooth [0.85–1.0] + instance noise decay + dropout → successful, FID −9.3%.
- **g4**: SN alone on D → D too constrained (negative result).
- **g8**: SN + MinibatchStd in the full g6 regularisation context → addressing mode collapse risk.

---

## References

[^1]: Radford, A., Metz, L., & Chintala, S. (2015). *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks*. arXiv:1511.06434.

[^2]: Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., & Hochreiter, S. (2017). *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium*. NeurIPS 2017.

[^3]: Bińkowski, M., Sutherland, D. J., Arbel, M., & Gretton, A. (2018). *Demystifying MMD GANs*. ICLR 2018.

[^4]: Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., & Bengio, Y. (2014). *Generative Adversarial Nets*. NeurIPS 2014.

[^5]: Karras, T., Laine, S., Aila, T., & Lehtinen, J. (2018). *Progressive Growing of GANs for Improved Quality, Stability, and Variation*. ICLR 2018.

[^6]: Ioffe, S., & Szegedy, C. (2015). *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*. ICML 2015.

[^7]: Odena, A., Dumoulin, V., & Olah, C. (2016). *Deconvolution and Checkerboard Artifacts*. Distill.

[^8]: Paszke, A., et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. NeurIPS 2019.

[^9]: Brock, A., Donahue, J., & Simonyan, K. (2019). *Large Scale GAN Training for High Fidelity Natural Image Synthesis*. ICLR 2019.

[^10]: Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., & Chen, X. (2016). *Improved Techniques for Training GANs*. NeurIPS 2016.

[^11]: Karras, T., Laine, S., & Aila, T. (2019). *A Style-Based Generator Architecture for Generative Adversarial Networks*. CVPR 2019.

[^12]: Miyato, T., Kataoka, T., Koyama, M., & Yoshida, Y. (2018). *Spectral Normalization for Generative Adversarial Networks*. ICLR 2018.

[^13]: Borji, A. (2022). *Pros and Cons of GAN Evaluation Measures: New Developments*. Computer Vision and Image Understanding.

[^14]: Liu, Z., Luo, P., Wang, X., & Tang, X. (2015). *Deep Learning Face Attributes in the Wild*. ICCV 2015.

[^15]: Sønderby, C. K., Caballero, J., Theis, L., Shi, W., & Huszár, F. (2016). *Amortised MAP Inference for Image Super-resolution*. arXiv:1610.04490.

[^16]: Roth, K., Lucchi, A., Nowozin, S., & Hofmann, T. (2017). *Stabilizing Training of Generative Adversarial Networks through Regularization*. NeurIPS 2017.
