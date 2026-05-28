# Axis 2 - Stronger damage prediction methods

## Current baseline

The current official damage champion is:

```text
unet_1024_long250_noleak_match_hist_all_aug-safe_sampler-damage-sqrt-alpha4_250epochs
```

Metrics:

- split: `splits_noleak_match_hist_all`
- epochs: `250`
- augmentation: `safe`
- sampler: `damage-sqrt`
- alpha: `4`
- mean IoU: `0.676624`
- damaged IoU: `0.446452`
- damaged precision: `0.605233`
- damaged recall: `0.629871`
- damaged F1: `0.617307`

The extra plain U-Net runs did not beat this model:

- alpha8, 250 epochs: damaged F1 `0.613364`, damaged IoU `0.442340`
- alpha4, 500 epochs: damaged F1 `0.613004`, damaged IoU `0.441965`

Conclusion: the current U-Net is a strong baseline. More plain U-Net sweeps are unlikely to be the best next use of GPU time unless new data or a new loss formulation appears.

## Immediate next step: TTA

Before retraining, evaluate the champion with test-time augmentation:

- no checkpoint change;
- no data split change;
- no training cost;
- useful signal about prediction stability under flips and rotations.

Recommended command:

```powershell
python scripts\evaluate_damage_tta.py `
  --checkpoint outputs\checkpoints\unet_1024_long250_noleak_match_hist_all_aug-safe_sampler-damage-sqrt-alpha4_250epochs\best_unet_portable.pt `
  --root data\raw\xbd\train `
  --split-csv data\processed\splits_noleak_match_hist_all\test_pairs.csv `
  --image-size 1024 `
  --batch-size 2 `
  --target-mode 3-class `
  --device cuda `
  --amp `
  --num-workers 0 `
  --tta-modes none flips rot90 d4 `
  --output-json outputs\predictions\damage_tta_champion_match_hist_all.json `
  --output-csv outputs\predictions\damage_tta_champion_match_hist_all.csv `
  --save-examples-dir outputs\figures\damage_tta_champion_match_hist_all `
  --num-examples 12
```

Metrics to inspect:

1. damaged IoU
2. damaged F1
3. damaged precision
4. damaged recall
5. mean IoU

If TTA improves damaged IoU/F1 without excessive recall/precision tradeoff, it can become a default evaluation/deployment option without retraining.

## Why architecture-level improvement is next

The plain local U-Net has now been tested across:

- image size 512 and 1024;
- several splits;
- class weights;
- CE-Dice;
- safe and damage-aware augmentation;
- weighted random sampling;
- 100, 250, and 500 epoch variants.

The best improvement now likely requires better representations rather than more small hyperparameter sweeps.

## Candidate order

### 1. Siamese U-Net

Purpose: explicitly compare pre-disaster and post-disaster images while keeping the segmentation workflow familiar.

Preferred design:

- shared encoder for pre and post images;
- feature fusion with `concat(pre, post, abs(post - pre))` or similar;
- decoder outputs the same 3 damage classes;
- compatible with 1024 images and batch size 2 on H100.

This is the most domain-aligned next architecture, but it should be implemented carefully rather than rushed.

### 2. U-Net with ResNet/EfficientNet encoder

Purpose: keep U-Net-like decoder behavior while improving encoder features.

First candidates:

- `smp_unet_effb3_6ch`
- `smp_unet_resnet50_6ch`

These can use 6-channel pre/post input directly with `encoder_weights=None`.

### 3. DeepLabV3+

Purpose: stronger context modeling and atrous spatial features.

First candidates:

- `smp_deeplabv3plus_resnet50_6ch`
- `smp_deeplabv3plus_effb3_6ch`

### 4. Attention U-Net later

Attention gates may help focus on building regions, but this is a second wave after establishing the SMP baselines and Siamese design.

### 5. SegFormer / ChangeFormer later

These are strong remote-sensing candidates, especially ChangeFormer for pre/post change detection, but they introduce a larger implementation and dependency surface. They should be reserved for a later, cleaner wave once Axis 2 baselines are established.

## Fair comparison protocol

All architecture candidates should use:

- no-leak splits;
- image size `1024`;
- target mode `3-class`;
- input semantics: pre RGB + post RGB, either 6-channel or Siamese two-stream;
- same metrics as the official baseline:
  - pixel accuracy;
  - mean IoU;
  - IoU per class;
  - precision/recall/F1 per class;
  - especially damaged IoU and damaged F1;
- first pass: `100 epochs`;
- finalists: `250 epochs`.

The first architecture sweep is defined in:

```text
configs/damage_arch_sweep_v1.csv
```

It is intentionally compact: six planned runs, not a broad 50-run campaign.

## Relationship with Building100

The Building100 segmentation sweep is running separately. Its goal is to find a stronger binary building mask model. Later, the best building segmenter can be combined with the best damage model through:

1. predicted-building clipping;
2. component majority damage smoothing;
3. future two-stage building segmentation + damage classification.

Axis 2 should therefore proceed in parallel:

- short-term: TTA for the current champion;
- next: architecture-level damage models;
- later: combine with the best building segmentation branch.

## Implementation note

Do not launch the architecture sweep until the model factory and training/evaluation scripts are smoke-tested. The Siamese U-Net should be implemented as a deliberate model module, not as a hurried edit to the existing `train_unet.py`.
