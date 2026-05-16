#!/usr/bin/env bash

set -euo pipefail
cd "${HOME}/work/CrisisMap-AI"

module --force purge
module load StdEnv/2023
module load python/3.11
module load gcc
module load arrow/23.0.1
module load cuda
module load opencv/4.13.0

source "${HOME}/virtualenvs/crisismap-ai/bin/activate"
export TRITON_CACHE_DIR="${HOME}/scratch/CrisisMap-AI/triton_cache"
mkdir -p "${TRITON_CACHE_DIR}" "${HOME}/scratch/CrisisMap-AI/run_logs"
mkdir -p outputs/checkpoints outputs/predictions

FORCE="${FORCE:-0}"
FORCE_INCOMPLETE="${FORCE_INCOMPLETE:-0}"
JOB_ID="${SLURM_JOB_ID:-manual}"

TARGET_SPLIT="${TARGET_SPLIT:?Set TARGET_SPLIT to a data/processed split directory name.}"
SPLIT_ALIAS="${SPLIT_ALIAS:-${TARGET_SPLIT#splits_noleak_}}"
PART_SUMMARY_CSV="${PART_SUMMARY_CSV:?Set PART_SUMMARY_CSV to the split-specific summary path.}"

IMAGE_SIZE="${IMAGE_SIZE:-1024}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LOSS="${LOSS:-ce-dice}"
CLASS_WEIGHTS="${CLASS_WEIGHTS:-0.05 1.0 4.0}"
LR="${LR:-1e-4}"
EPOCHS="${EPOCHS:-100}"
AUGMENT_PROB="${AUGMENT_PROB:-0.5}"
DAMAGE_AUGMENT_THRESHOLD="${DAMAGE_AUGMENT_THRESHOLD:-0.001}"
HIGH_DAMAGE_THRESHOLD="${HIGH_DAMAGE_THRESHOLD:-0.06}"
SUMMARY_CSV="outputs/predictions/unet_1024_noleak_aug_sampler_100epochs_summary.csv"

COMBOS=(
  "none|none|4"
  "safe|none|4"
  "damage-aware|none|4"
  "none|damage-simple|4"
  "safe|damage-simple|4"
  "damage-aware|damage-simple|4"
  "safe|damage-sqrt|4"
  "damage-aware|damage-sqrt|4"
)

split_has_files() {
  local split_dir="$1"
  [ -f "data/processed/${split_dir}/train_pairs.csv" ] &&
    [ -f "data/processed/${split_dir}/val_pairs.csv" ] &&
    [ -f "data/processed/${split_dir}/test_pairs.csv" ]
}

ensure_split() {
  if [ ! -f "data/processed/splits_full/test_pairs.csv" ]; then
    echo "Missing common test split: data/processed/splits_full/test_pairs.csv" >&2
    return 1
  fi
  if ! split_has_files "$TARGET_SPLIT"; then
    echo "Missing split files: data/processed/${TARGET_SPLIT}" >&2
    echo "Create advanced no-leak splits before launching this job." >&2
    return 1
  fi
}

history_complete() {
  local history_json="$1"
  local expected_epochs="$2"
  [ -f "$history_json" ] || return 1
  python - "$history_json" "$expected_epochs" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = int(sys.argv[2])
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    sys.exit(1)
history = data if isinstance(data, list) else data.get("history") if isinstance(data, dict) else None
if not isinstance(history, list) or len(history) < expected:
    sys.exit(1)
epochs = []
for item in history:
    if isinstance(item, dict) and "epoch" in item:
        try:
            epochs.append(int(item["epoch"]))
        except (TypeError, ValueError):
            pass
if epochs and max(epochs) < expected:
    sys.exit(1)
sys.exit(0)
PY
}

alpha_label() {
  local alpha="$1"
  if [[ "$alpha" == *.* ]]; then
    python - "$alpha" <<'PY'
import sys
value = float(sys.argv[1])
if value.is_integer():
    print(int(value))
else:
    print(str(sys.argv[1]).replace(".", "p"))
PY
  else
    echo "$alpha"
  fi
}

experiment_name() {
  local augment_mode="$1"
  local sampler="$2"
  local alpha="$3"
  local name
  name="unet_${IMAGE_SIZE}_aug_noleak_${SPLIT_ALIAS}_${EPOCHS}epochs_aug-${augment_mode}_sampler-${sampler}"
  if [ "$sampler" = "damage-sqrt" ]; then
    name="${name}-alpha$(alpha_label "$alpha")"
  fi
  echo "$name"
}

rebuild_summary() {
  python scripts/rebuild_noleak_aug_sampler_summary.py \
    --output "$SUMMARY_CSV" \
    --image-size "$IMAGE_SIZE" \
    --batch-size "$BATCH_SIZE" \
    --loss "$LOSS" \
    --class-weights "$CLASS_WEIGHTS" \
    --lr "$LR" \
    --epochs "$EPOCHS" \
    --augment-prob "$AUGMENT_PROB" \
    --damage-augment-threshold "$DAMAGE_AUGMENT_THRESHOLD" \
    --high-damage-threshold "$HIGH_DAMAGE_THRESHOLD"

  python scripts/rebuild_noleak_aug_sampler_summary.py \
    --output "$PART_SUMMARY_CSV" \
    --splits "$TARGET_SPLIT" \
    --image-size "$IMAGE_SIZE" \
    --batch-size "$BATCH_SIZE" \
    --loss "$LOSS" \
    --class-weights "$CLASS_WEIGHTS" \
    --lr "$LR" \
    --epochs "$EPOCHS" \
    --augment-prob "$AUGMENT_PROB" \
    --damage-augment-threshold "$DAMAGE_AUGMENT_THRESHOLD" \
    --high-damage-threshold "$HIGH_DAMAGE_THRESHOLD"
}

train_and_eval() {
  local augment_mode="$1"
  local sampler="$2"
  local alpha="$3"
  local name
  local output_dir
  local history_json
  local checkpoint
  local metrics_json
  local weights_array

  name="$(experiment_name "$augment_mode" "$sampler" "$alpha")"
  output_dir="outputs/checkpoints/${name}"
  history_json="${output_dir}/metrics_history.json"
  checkpoint="${output_dir}/best_unet.pt"
  metrics_json="outputs/predictions/${name}_test_metrics.json"
  read -r -a weights_array <<< "$CLASS_WEIGHTS"

  echo
  echo "==> ${name}"
  echo "split=${TARGET_SPLIT} augment_mode=${augment_mode} sampler=${sampler} alpha=${alpha}"

  if [ "$FORCE" = "1" ]; then
    rm -rf -- "$output_dir"
    rm -f -- "$metrics_json"
  elif history_complete "$history_json" "$EPOCHS" && [ -f "$metrics_json" ]; then
    echo "Complete run found; skipping."
    return 0
  elif [ -d "$output_dir" ]; then
    if history_complete "$history_json" "$EPOCHS" && [ -f "$checkpoint" ]; then
      echo "Training complete; evaluating missing metrics."
    elif [ "$FORCE_INCOMPLETE" = "1" ]; then
      echo "FORCE_INCOMPLETE=1: removing incomplete output before retraining."
      rm -rf -- "$output_dir"
      rm -f -- "$metrics_json"
    else
      echo "WARNING: Incomplete output folder, not evaluating partial checkpoint: $output_dir"
      echo "WARNING: Relaunch with FORCE_INCOMPLETE=1 to retrain only incomplete runs."
      return 0
    fi
  fi

  if ! history_complete "$history_json" "$EPOCHS"; then
    python -m src.crisismap.training.train_unet \
      --root data/raw/xbd/train \
      --train-csv "data/processed/${TARGET_SPLIT}/train_pairs.csv" \
      --val-csv "data/processed/${TARGET_SPLIT}/val_pairs.csv" \
      --output-dir "$output_dir" \
      --image-size "$IMAGE_SIZE" \
      --batch-size "$BATCH_SIZE" \
      --epochs "$EPOCHS" \
      --target-mode 3-class \
      --loss "$LOSS" \
      --class-weights "${weights_array[@]}" \
      --lr "$LR" \
      --num-workers 8 \
      --augment-mode "$augment_mode" \
      --augment-prob "$AUGMENT_PROB" \
      --damage-augment-threshold "$DAMAGE_AUGMENT_THRESHOLD" \
      --sampler "$sampler" \
      --damage-sampling-alpha "$alpha" \
      --high-damage-threshold "$HIGH_DAMAGE_THRESHOLD"
  fi

  if ! history_complete "$history_json" "$EPOCHS" || [ ! -f "$checkpoint" ]; then
    echo "ERROR: Run did not complete cleanly; refusing to evaluate partial checkpoint." >&2
    return 1
  fi

  if [ ! -f "$metrics_json" ]; then
    python -m src.crisismap.evaluation.evaluate_unet \
      --root data/raw/xbd/train \
      --split-csv data/processed/splits_full/test_pairs.csv \
      --checkpoint "$checkpoint" \
      --output "$metrics_json" \
      --image-size "$IMAGE_SIZE" \
      --batch-size 1 \
      --target-mode 3-class \
      --num-workers 8
  fi
}

ensure_split

echo "No-leak augmentation/sampler campaign"
echo "TARGET_SPLIT=${TARGET_SPLIT} SPLIT_ALIAS=${SPLIT_ALIAS}"
echo "IMAGE_SIZE=${IMAGE_SIZE} BATCH_SIZE=${BATCH_SIZE} LOSS=${LOSS} LR=${LR} EPOCHS=${EPOCHS}"
echo "AUGMENT_PROB=${AUGMENT_PROB} DAMAGE_AUGMENT_THRESHOLD=${DAMAGE_AUGMENT_THRESHOLD}"
echo "HIGH_DAMAGE_THRESHOLD=${HIGH_DAMAGE_THRESHOLD}"
echo "FORCE=${FORCE} FORCE_INCOMPLETE=${FORCE_INCOMPLETE}"
echo "Global summary: ${SUMMARY_CSV}"
echo "Part summary: ${PART_SUMMARY_CSV}"

for combo in "${COMBOS[@]}"; do
  IFS="|" read -r augment_mode sampler alpha <<< "$combo"
  name="$(experiment_name "$augment_mode" "$sampler" "$alpha")"
  run_log="${HOME}/scratch/CrisisMap-AI/run_logs/${name}-${JOB_ID}.log"
  train_and_eval "$augment_mode" "$sampler" "$alpha" 2>&1 | tee "$run_log"
  rebuild_summary
done

rebuild_summary
