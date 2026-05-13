# Project Journal — DeepFakeFace (M.IA003)

One row per change. Fill **Result** and **Decision** as each iteration runs;
the values come straight from `outputs/resnet_iterations.csv`. The slides
should cite this file directly (PDF §3.11).

## Classifier ([resnetExploration.ipynb](resnetExploration.ipynb))

| # | Iteration | Hypothesis | Change | Expected | Result (F1 / bal_acc / AUC / ECE) | Decision |
|---|-----------|------------|--------|----------|-----------------------------------|----------|
| v0 | `baseline_scratch` | Need an anchor before tuning anything | ResNet-18 from scratch, no aug, wiki + inpainting | F1 ~0.90, miscalibrated | _fill_ | accept as floor |
| v1 | `pretrained` | ImageNet features should transfer for free | Switch to `ResNet18_Weights.DEFAULT`, freeze stem 3 ep | +F1, lower ECE | _fill_ | _fill_ |
| v2 | `+augmentation` | v1 may shortcut on JPEG / scale cues | RandomResizedCrop, flip, jitter, blur, JPEG re-encode | tiny F1 dip, big OOD lift | _fill_ | _fill_ |
| v3 | `+all_generators` | Brief asks for all 3 generators | Add sd1.5 + insightface to training | F1 dips, balanced acc up | _fill_ | _fill_ |
| v4 | `resnet34` | Maybe ResNet-18 saturates on 4 sources | Same recipe, deeper backbone | small lift, more compute | _fill_ | _fill_ |
| v5 | `+heavy_dropout` *(neg)* | Over-regularising should hurt | dropout=0.5 ablation | F1 drops | confirm hypothesis → keep v3 setting |
| §6 | LOO OOD | Quantify generator-specific shortcuts | Train on 2 fakes, test on the 3rd | F1 drops 5–15 pts | _fill_ |
| v6 | `optuna_winner` | Tune the v3 recipe properly | 40×15 (or 100×30) Optuna on **val** | best F1, single test eval | _fill_ |

## Generator ([ganImprove.ipynb](ganImprove.ipynb)) — backlog

Tracked separately. PDF §2.2 priorities: FID curve, EMA, spectral-norm D,
DiffAugment, full 30k real images, latent interpolations.

| # | Iteration | Change | FID @ end | Notes |
|---|-----------|--------|-----------|-------|
| g0 | baseline | current code (folds 0–5, no FID) | _fill_ | establish floor |
| g1 | + FID logging | torch-fidelity every 25 epochs | _fill_ | required for any further claim |
| g2 | + EMA | 0.999 EMA on G | _fill_ | typically –5/–15 FID |
| g3 | + spectral-norm D | replace BN(D) with SN | _fill_ | combats mode collapse |
| g4 | full 30k | folds 0–99 | _fill_ | 10× data |

## Open risks / TODO before May 15

- [ ] Run v0–v6 end-to-end on the training machine (full epoch budgets)
- [ ] Capture screenshots of `outputs/iteration_evolution.png` and `loo_ood_chart.png` for slides
- [ ] Confirm all three group members can speak to all three sections (course requirement)
- [ ] Fill the auto-evaluation Moodle file
- [ ] Pin versions in `requirements.txt` matching the run that produced final numbers
