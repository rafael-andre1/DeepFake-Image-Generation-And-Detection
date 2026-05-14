# GAN Evolution Guide — From Baseline to Production-Ready Generator

> **Scope:** This document traces every incremental design decision made across nine model versions (`g0` → `g8`), explaining *why* each change was introduced, what problem it addressed, and which papers support it. The discussion is organised thematically, not strictly chronologically, so each section can be read independently.

---

## Quick Reference — Version Snapshot Table

| Version | Epochs | Folds (data scale) | Key Changes vs. Previous |
|---|---|---|---|
| **g0** | 200 | 0–5 | Baseline DCGAN; full instrumentation |
| **g1** | 200 | 0–20 | ↑ dataset size (4× more folds) |
| **g2** | 200 | 0–20 | ↓ LR_D (`1e-4 → 5e-5`) + label smoothing (real = 0.9) |
| **g3** | 200 | 0–20 | + EMA on generator (decay = 0.999) |
| **g4** | 200 | 0–20 | + SpectralNorm on D, BatchNorm removed from D |
| **g5** | 200 | 0–99 | ↑↑ dataset (all folds); best-FID checkpoint saved |
| **g6** | 200 | 0–99 | + Face-crop preprocessing; + instance noise; + dropout in D; one-sided label smoothing |
| **g7** | **400** | 0–5 | ↑↑ epochs (2×); keeps g6 regularisation stack; saves every-epoch JSONL |
| **g8** | 400 | 0–5 | Discriminator redesigned: SpectralNorm + MinibatchStd layer |

---

## 1. Architecture Foundations — What Never Changed

All versions share the same **DCGAN generator** (Radford et al., 2015) and a latent vector of dimension **100**, projected from `z ∈ ℝ^{100}` to a 4×4 spatial map via `ConvTranspose2d`, then upsampled four times through a Upsample-Conv-BN-ReLU stack to produce **64 × 64** RGB images with `Tanh` output. The fixed output resolution of 64 × 64 was a deliberate budget choice: it keeps VRAM manageable and training times tractable, while being sufficient to evaluate perceptual quality with FID/KID.

```
z (100,1,1) → ConvTranspose2d → BN → ReLU (4×4)
             → Upsample → Conv → BN → ReLU (8×8)
             → Upsample → Conv → BN → ReLU (16×16)
             → Upsample → Conv → BN → ReLU (32×32)
             → Upsample → Conv → Tanh           (64×64)
```

The **Upsample + Conv** pattern (rather than the strided `ConvTranspose2d` used in classic DCGAN) avoids the checkerboard artefacts caused by uneven overlap in transpose convolutions — a well-documented failure mode described by Odena et al. (2016) *"Deconvolution and Checkerboard Artifacts"*.

The **loss function** is `BCEWithLogitsLoss` throughout all versions. Using raw logits instead of sigmoid outputs and then BCE is numerically more stable because it fuses the sigmoid into the log-sum-exp computation (PyTorch documentation). This is consistent across all nine versions.

---

## 2. Image Size

**Fixed at 64 × 64 throughout all versions.**

The original DCGAN paper (Radford et al., 2015) validated the architecture at this resolution. Scaling to 128 × 128 or higher would require additional upsampling stages in the generator, additional downsampling in the discriminator, and roughly 4× the VRAM and training time per step — not justified here because the dataset itself (Wikipedia face photographs) is heterogeneous in pose, lighting, and cropping, making a well-trained 64 × 64 model a more achievable first target.

The `dcganFormat(IMAGE_SIZE)` transform applies:
- `CenterCrop` → `Resize(64)` → `ToTensor` → `Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])`

This maps pixel values to `[-1, 1]`, matching the `Tanh` output range of the generator. Using values outside `[-1, 1]` as supervision signal while Tanh saturates at the boundary would create systematic gradient mismatch.

**g6** introduces a smarter preprocessing: an **OpenCV Haar-cascade face detector** crops around the largest detected face (with a 45 % margin) before resizing to 64 × 64. When no face is detected, it falls back to a centre-square crop. This changes the *effective* content of each 64 × 64 pixel, even though the tensor resolution stays identical. The motivation is that standard DCGAN at 64 × 64 struggles with the full compositional variety of Wikipedia photos (different backgrounds, body proportions, lighting) — focusing on the face simplifies the data distribution the generator needs to learn. This is the same principle behind Karras et al.'s progressive growing (ProGAN, 2018): reducing distributional complexity first.

---

## 3. Epochs

### g0 → g6: 200 epochs

200 epochs was chosen as the initial budget. For a dataset of ~hundreds to a few thousand images (folds 0–5) with batch size 64, 200 epochs represents tens of thousands of gradient steps — sufficient to see stable convergence curves and meaningful FID readings at every 50-epoch checkpoint.

### g7 and g8: 400 epochs (2× increase)

After the global analysis recorded in g5–g6, it was clear that the generator quality kept improving well past epoch 100 and had not fully plateaued by epoch 200. The justification for doubling the budget is:

> *GAN generators tend to improve monotonically in FID up to a point, after which mode collapse or discriminator saturation can cause regression. Running longer with the g6 regularisation stack (dropout + instance noise) should delay the saturation point.*

This is consistent with Brock et al. (BigGAN, 2019), who found that larger batch sizes and longer training schedules are the most reliable levers for GAN quality. Here we approximate that insight with longer training rather than larger batches (VRAM constraint).

**g7 and g8 also introduce per-epoch JSONL snapshots** of the generator output, stored in calibrated split files of ~90 MB each. The split logic computes `EPOCHS_PER_SPLIT = TARGET_BYTES // frame_bytes` at runtime, so the file size stays within bounds regardless of epoch count. This enables visual inspection of the full training trajectory without manual cherry-picking of snapshots.

---

## 4. Metrics

### 4.1 Generator Losses: D(x), D(G(z))

All versions track three adversarial health signals per epoch:

| Signal | Meaning | Healthy range |
|---|---|---|
| `D(x)` | Average discriminator probability on real images | Should stay < 1.0; ideally ~0.5–0.8 |
| `D(G(z)) fake` | D probability on fakes (during D update) | Should be low early, then rise |
| `D(G(z)) gen` | D probability on fakes (during G update) | Should rise as G improves |

If `D(x) → 1` and `D(G(z)) fake → 0` simultaneously, the discriminator has saturated: it assigns perfect confidence to both classes, and the generator receives near-zero gradients. This was the exact failure mode observed in g5, which motivated the g6 regularisation overhaul.

### 4.2 Fréchet Inception Distance (FID)

FID (Heusel et al., 2017 — *"GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium"*) is the primary evaluation metric. It measures the Fréchet distance between two multivariate Gaussians fitted to the InceptionV3 pool3 features of real and generated images:

```
FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2(Σ_r Σ_g)^{1/2})
```

Lower is better. Key properties:
- Sensitive to both **image quality** (mode collapse → higher FID) and **diversity** (distribution shift → higher FID).
- Requires a sufficiently large sample to be statistically reliable; the implementations here use `FID_NUM_IMAGES = 300`, which is a trade-off between measurement cost and reliability. The original paper recommends ≥ 10,000 images; 300 gives noisier but directionally useful readings.

FID is computed every 50 epochs (`FID_EVERY = 50`) to avoid the substantial overhead of running InceptionV3 over hundreds of batches at every epoch.

### 4.3 Kernel Inception Distance (KID)

KID (Bińkowski et al., 2018 — *"Demystifying MMD GANs"*) is a companion metric with two advantages over FID:

1. **Unbiased estimator**: FID has a sample-size-dependent bias; KID is an unbiased MMD estimate.
2. **Valid with small samples**: KID remains meaningful at `n = 50` (used here as `subset_size`), whereas FID degrades significantly below ~1000 images.

```
KID = MMD²(InceptionV3(real), InceptionV3(fake))
```

In practice, KID is used here as a confirmation metric alongside FID. When both FID and KID trend downward, the improvement is considered robust.

### 4.4 Best-FID Checkpoint (g5 onward)

Starting in g5, a `best_fid_checkpoint.pt` is written automatically whenever a new FID minimum is observed. This decouples **model selection** from **training duration**: the model that minimises FID is not necessarily the one from the final epoch, especially since GAN generators can partially degrade at late epochs due to memorisation or discriminator saturation. The rationale echoes early-stopping in supervised learning (Goodfellow et al., *Deep Learning* textbook, Chapter 7).

---

## 5. Optimizations

### 5.1 Adam Optimizer — Fixed Throughout

All versions use `Adam(β₁=0.5, β₂=0.999)` for both generator and discriminator. This deviates from the default `β₁=0.9` recommended by Kingma & Ba (2015) — the original DCGAN paper (Radford et al., 2015) empirically found `β₁=0.5` to be more stable for GAN training. The intuition is that a lower `β₁` reduces the momentum contribution of the first moment, preventing the optimizer from overshooting the saddle-point equilibrium that GANs converge towards.

### 5.2 Learning Rate Schedule

| Period | LR_G | LR_D | Rationale |
|---|---|---|---|
| g0–g1 | 2e-4 | 1e-4 | Baseline DCGAN rates (Radford 2015) |
| g2 | 2e-4 | **5e-5** | Attempt to slow D saturation by halving its LR |
| g3–g8 | 2e-4 | 1e-4 | Reverted after g2 showed worse FID |

**Why 2:1 ratio (G vs D)?** The generator and discriminator play a zero-sum game. If D learns much faster than G, it saturates and stops providing useful gradient signal to G. Keeping `LR_G > LR_D` gives the generator a slight advantage, consistent with the two-time-scale update rule (TTUR) of Heusel et al. (2017), who show that different learning rates can facilitate convergence to a local Nash equilibrium.

**Why g2 failed with `LR_D = 5e-5`?** Slowing down D too aggressively means it can no longer distinguish real from fake quickly enough, so the generator receives an easier training signal but one that is less informative. The resulting FID was higher than g1, so the rate was restored in g3.

### 5.3 EMA on Generator — g3 onward

```python
# Exponential Moving Average update (per batch)
ema_param = decay * ema_param + (1 - decay) * param
# decay = 0.999
```

EMA maintains a **shadow copy** of the generator (`netG_ema`) whose weights are a weighted average of all past parameter states. The key insight (Polyak & Juditsky, 1992; also used in Karras et al. ProGAN 2018 and StyleGAN 2019) is that SGD-based optimizers oscillate around the loss minimum — EMA follows the *envelope* of those oscillations, producing weights that lie closer to the true minimum than any individual iterate.

For GANs specifically, the benefit is reduced sample-to-sample variance: the EMA generator produces more visually consistent images at any given checkpoint, which is also reflected in lower FID because the Inception features are more tightly clustered.

From g3 onward, **all evaluations (FID, KID, PNG samples, latent interpolations) use `netG_ema`**, not `netG`. The training itself continues with `netG`; `netG_ema` is never back-propagated through.

---

## 6. Regularizations

### 6.1 Data Scale — g1

The first and most impactful change: expanding folds from `[0, 5]` to `[0, 20]` (4× more images). This is not a regularization in the classical sense, but data volume is the most direct counter to overfitting and mode collapse in GANs. With only ~5 folds, the discriminator memorises the training set, causing it to saturate early. More real images force D to learn more general features, providing better gradient signal to G. This was confirmed to be the largest single improvement in FID.

### 6.2 Label Smoothing — g2, g6, g7, g8

**One-sided label smoothing** replaces hard real labels (`1.0`) with soft targets:

- **g2**: fixed soft label of `0.9` for real images.
- **g6/g7/g8**: stochastic soft label uniformly sampled from `[0.85, 1.0]` per batch.

Salimans et al. (2016) — *"Improved Techniques for Training GANs"* — introduced one-sided label smoothing specifically to prevent the discriminator from assigning extreme log-probabilities to real images, which causes the cross-entropy gradients to the generator to vanish. By replacing `target = 1.0` with `target ∈ [0.85, 1.0]`, the discriminator is penalised for being overconfident on real samples, preserving non-zero gradient magnitude for the generator.

Only real labels are smoothed (one-sided). Smoothing fake labels toward 0 is not done because it would give G incorrect signal: it would encourage G to produce images that D rates as "slightly real" rather than "fully real".

### 6.3 Instance Noise — g6, g7, g8

```python
# Decaying Gaussian noise added to both real and fake images before D
sigma(epoch) = sigma_0 * max(0, 1 - epoch / T_decay)
# sigma_0 = 0.05, T_decay = 80
```

Instance noise (Sønderby et al., 2017 — *"Amortised MAP Inference for Image Super-resolution"*, and more directly Jenni & Favaro, 2019) injects small Gaussian noise onto the discriminator's inputs. This has several effects:

1. **Prevents D saturation early in training** — noisy inputs make classification harder, so D cannot immediately achieve perfect separation.
2. **Smooths the discriminator's decision boundary** — conceptually equivalent to a data augmentation that forces D to learn features robust to small pixel perturbations.
3. **Decays to zero** — as training progresses and G becomes better, the noise is annealed away so D can make fine-grained distinctions. The 80-epoch linear decay schedule was chosen so noise is fully gone by ~40% through training.

The same noise is applied to real and fake inputs (symmetric), so neither class is systematically disadvantaged.

### 6.4 Dropout in Discriminator — g6, g7, g8

```python
nn.Dropout2d(p=0.20)   # after first three conv blocks
```

Dropout2d (Tompson et al., 2015) zeros entire feature map channels during training, acting as a structural regulariser that prevents co-adaptation of discriminator features. In the context of GANs, this directly limits how quickly D can memorise the real training set, keeping the adversarial game competitive longer.

The dropout rate of `p = 0.20` is mild — enough to slow D down without making it unable to learn useful features. It is applied after the first three convolution blocks but **not** after the last, to preserve the signal quality at the final scoring layer. Dropout is automatically disabled during `model.eval()` calls (used for FID/KID and sample generation).

### 6.5 Spectral Normalization (SpectralNorm) — g4 first attempt, g8 final

Miyato et al. (2018) — *"Spectral Normalization for Generative Adversarial Networks"* — propose constraining the Lipschitz constant of the discriminator by normalising each weight matrix by its largest singular value:

```
W_SN = W / sigma(W)   where sigma(W) = largest singular value of W
```

This constrains `||D||_Lip <= 1` globally, which provides a theoretical stability guarantee: the discriminator cannot grow arbitrarily sharp gradients that would swamp the generator.

**g4 — First attempt (reverted):** SpectralNorm was applied alongside EMA (g3 settings), and BatchNorm was removed from D. The result was *worse* FID than g3. The hypothesis is that SpectralNorm + no BatchNorm made D too weak to provide meaningful gradients in the early epochs, effectively giving G an easy task that it couldn't learn to generalise from.

**g8 — Final implementation:** SpectralNorm is combined with the full g6 regularisation stack (instance noise + one-sided label smoothing + Dropout2d). The discriminator retains `bias=True` (which was `False` in earlier versions), which is correct when SpectralNorm replaces BatchNorm — bias terms are needed to allow the layer to shift its activation distribution. The architecture also gains a **MinibatchStd** layer.

### 6.6 MinibatchStd Layer — g8

```python
class MinibatchStd(nn.Module):
    def forward(self, x):
        std = x.std(dim=0, keepdim=True)      # (1, C, H, W)
        std = std.mean(dim=1, keepdim=True)    # (1, 1, H, W)
        std = std.expand(x.size(0), ...)
        return torch.cat([x, std], dim=1)      # appended as extra channel
```

MinibatchStd was introduced by Karras et al. (2018) in **ProGAN** (*"Progressive Growing of GANs for Improved Quality, Stability, and Variation"*). The idea: appending the batch-level standard deviation as an extra feature map gives the discriminator a signal about **sample diversity within the batch**. If the generator produces low-diversity outputs (mode collapse), the std feature map will be nearly zero and D can learn to flag this. If G produces diverse outputs, the std will be larger, matching real data statistics.

This directly attacks mode collapse without making D stronger at classifying individual images. It is inserted between the feature extractor and the final conv/classifier head:

```
Input → features (4x4, ndf*8 channels)
      → MinibatchStd → (4x4, ndf*8+1 channels)
      → SN(Conv2d → 1x1x1) → scalar score
```

In g8, the final `Conv2d` is also wrapped in SpectralNorm so the Lipschitz constraint applies all the way to the output.

---

## 7. Ablation Summary — What Worked vs. What Didn't

| Change | Introduced in | Outcome | Retained? |
|---|---|---|---|
| More data (4x folds) | g1 | Largest FID improvement | Yes (g1–g6) |
| Reduced LR_D (5e-5) | g2 | Worse FID; D too weak | Reverted in g3 |
| Label smoothing (fixed 0.9) | g2 | Ambiguous alone; useful when combined | Evolved in g6 |
| EMA on generator | g3 | More stable samples; slight FID gain | Yes (g3–g8) |
| SpectralNorm D, no BN | g4 | Worse FID in isolation | Reverted in g5; reintroduced with full stack in g8 |
| Full data (all 99 folds) | g5 | Best single-model FID | Yes (g5–g6) |
| Best-FID checkpoint saving | g5 | Better model selection | Yes (g5–g8) |
| Face crop preprocessing | g6 | Cleaner distribution | Yes (g6) |
| Instance noise + decay | g6 | Prevents early D saturation | Yes (g6–g8) |
| Dropout in D | g6 | Limits D memorisation | Yes (g6–g8) |
| Stochastic label smoothing | g6 | More effective than fixed smoothing | Yes (g6–g8) |
| 2x epoch budget | g7–g8 | Quality keeps improving | Yes (g7–g8) |
| MinibatchStd + SN (full stack) | g8 | Theoretically best D setup | Current best |

---

## 8. References

| Paper | Key Contribution | Used In |
|---|---|---|
| Goodfellow et al. (2014). *Generative Adversarial Nets.* NeurIPS. | Original GAN framework | All versions |
| Radford et al. (2015). *Unsupervised Representation Learning with Deep Convolutional GANs (DCGAN).* ICLR 2016. | Architecture template, Adam β₁=0.5 | All versions |
| Salimans et al. (2016). *Improved Techniques for Training GANs.* NeurIPS. | Label smoothing, feature matching | g2, g6–g8 |
| Odena et al. (2016). *Deconvolution and Checkerboard Artifacts.* Distill. | Upsample+Conv over ConvTranspose2d | All versions (generator) |
| Heusel et al. (2017). *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium.* NeurIPS. | FID metric; TTUR learning rates | All versions (evaluation); LR choices |
| Miyato et al. (2018). *Spectral Normalization for Generative Adversarial Networks.* ICLR. | SpectralNorm on discriminator | g4, g8 |
| Bińkowski et al. (2018). *Demystifying MMD GANs.* ICLR. | KID metric | All versions (evaluation) |
| Karras et al. (2018). *Progressive Growing of GANs (ProGAN).* ICLR. | MinibatchStd layer; EMA weights | g8 (MinibatchStd); g3–g8 (EMA) |
| Karras et al. (2019). *A Style-Based Generator Architecture for GANs (StyleGAN).* CVPR. | EMA on generator weights | g3–g8 |
| Kingma & Ba (2015). *Adam: A Method for Stochastic Optimization.* ICLR. | Adam optimizer | All versions |
| Polyak & Juditsky (1992). *Acceleration of Stochastic Approximation by Averaging.* | Theoretical basis for EMA/weight averaging | g3–g8 |
| Sønderby et al. (2017) / Jenni & Favaro (2019). | Instance noise for GAN stability | g6–g8 |
