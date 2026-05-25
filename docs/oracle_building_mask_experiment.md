# Oracle building-mask experiment

## Purpose

This experiment estimates how much the current 3-class U-Net could improve if
building segmentation were perfect.

The current task predicts:

- `0`: background
- `1`: no damage / intact building
- `2`: damaged building

In practice, the model must solve two coupled problems at once:

1. separate buildings from background;
2. decide whether each building is intact or damaged.

The oracle experiment isolates the first problem by using the ground-truth
building mask during evaluation only. It is not a deployable method and it must
not be used during training.

## Evaluation modes

### raw

The normal model prediction is evaluated unchanged.

### oracle_building_clip

The ground-truth building mask is computed as:

```text
building_gt = target > 0
```

Every predicted pixel outside `building_gt` is forced to background `0`.
Predictions inside the ground-truth building mask are kept unchanged.

This measures how many errors are caused by predicting buildings in impossible
background regions.

### oracle_building_component_majority

This is the main oracle mode.

The prediction is first clipped using the ground-truth building mask. Then each
connected component of the ground-truth building mask is treated as one building
instance. For each component:

- only predicted classes `1` and `2` are counted;
- if class `2` is the majority, the full component is labeled damaged;
- otherwise, the full component is labeled no damage;
- if no class `1` or `2` pixel is predicted inside the component, the default
  fallback is class `1` (`no_damage`).

The script also supports `--empty-component-policy gt_majority`, which uses the
dominant ground-truth building class as fallback for empty components. The
default `no_damage` fallback is more conservative.

## Oracle assumption

In xBD/xView2, each building instance is generally associated with one damage
level. Therefore, forcing a single predicted damage class per ground-truth
building component is a meaningful oracle approximation.

This does not mean that the final model should receive ground-truth building
masks. It only estimates the potential benefit of a future two-stage or
multi-task pipeline.

## How to interpret results

A strong gain in damaged IoU or damaged F1 suggests that the current U-Net is
losing performance because building localization and damage prediction are
entangled. In that case, a future architecture could benefit from:

- an explicit building segmentation head;
- a per-building damage classifier;
- post-processing that enforces coherent labels per building;
- a stronger change-detection or instance-aware model.

A weak gain means that perfect building localization alone would not solve the
main error modes. It does not invalidate two-stage or multi-task approaches,
because the model may still need better temporal features, richer context, more
balanced sampling, or a stronger backbone.

## Relation to a future pipeline

The oracle approximates this future direction:

1. segment buildings;
2. classify each building, or each building pixel, as intact or damaged;
3. optionally map predictions back to geographic coordinates using xBD
   geotransform metadata.

The current script only evaluates the theoretical upper-bound effect of step 1
using ground truth.

## Example command

```powershell
python scripts/evaluate_oracle_building_mask_gain.py `
  --root data/raw/xbd/train `
  --split-csv data/processed/splits_full/test_pairs.csv `
  --checkpoint outputs/checkpoints/unet_512_ce_dice_w005_1_4_50epochs/best_unet.pt `
  --image-size 512 `
  --batch-size 2 `
  --device cpu `
  --output-json outputs/predictions/oracle_building_mask_gain.json `
  --output-csv outputs/predictions/oracle_building_mask_gain.csv `
  --save-examples-dir outputs/figures/oracle_building_mask_gain `
  --num-examples 8
```

For a quick local smoke test:

```powershell
python scripts/evaluate_oracle_building_mask_gain.py `
  --root data/raw/xbd/train `
  --split-csv data/processed/splits_full/test_pairs.csv `
  --checkpoint outputs/checkpoints/unet_512_ce_dice_w005_1_4_50epochs/best_unet.pt `
  --image-size 256 `
  --batch-size 1 `
  --max-samples 2 `
  --device cpu `
  --output-json outputs/predictions/oracle_smoke.json
```
