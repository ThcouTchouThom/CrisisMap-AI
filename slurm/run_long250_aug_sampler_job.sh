#!/usr/bin/env bash

set -euo pipefail
cd "${HOME}/work/CrisisMap-AI"
export PYTHONPATH="${PWD}/src:${PWD}:${PYTHONPATH:-}"

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
AUGMENT_MODE="${AUGMENT_MODE:?Set AUGMENT_MODE.}"
SAMPLER="${SAMPLER:?Set SAMPLER.}"

IMAGE_SIZE="${IMAGE_SIZE:-1024}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LOSS="${LOSS:-ce-dice}"
CLASS_WEIGHTS="${CLASS_WEIGHTS:-0.05 1.0 4.0}"
LR="${LR:-1e-4}"
EPOCHS="${EPOCHS:-250}"
AUGMENT_PROB="${AUGMENT_PROB:-0.5}"
DAMAGE_AUGMENT_THRESHOLD="${DAMAGE_AUGMENT_THRESHOLD:-0.001}"
DAMAGE_SAMPLING_ALPHA="${DAMAGE_SAMPLING_ALPHA:-4}"
HIGH_DAMAGE_THRESHOLD="${HIGH_DAMAGE_THRESHOLD:-0.06}"
SUMMARY_CSV="outputs/predictions/unet_1024_long250_aug_sampler_summary.csv"

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
  local name
  name="unet_${IMAGE_SIZE}_long250_noleak_${SPLIT_ALIAS}_aug-${AUGMENT_MODE}_sampler-${SAMPLER}"
  if [ "$SAMPLER" = "damage-sqrt" ]; then
    name="${name}-alpha$(alpha_label "$DAMAGE_SAMPLING_ALPHA")"
  fi
  echo "${name}_${EPOCHS}epochs"
}

rebuild_summary() {
  python scripts/rebuild_long250_aug_sampler_summary.py \
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
}

run_job() {
  local name
  local output_dir
  local history_json
  local checkpoint
  local metrics_json
  local weights_array

  name="$(experiment_name)"
  output_dir="outputs/checkpoints/${name}"
  history_json="${output_dir}/metrics_history.json"
  checkpoint="${output_dir}/best_unet.pt"
  metrics_json="outputs/predictions/${name}_test_metrics.json"
  read -r -a weights_array <<< "$CLASS_WEIGHTS"

  echo "Long250 augmentation/sampler job"
  echo "Experiment: ${name}"
  echo "split=${TARGET_SPLIT} augment_mode=${AUGMENT_MODE} sampler=${SAMPLER} alpha=${DAMAGE_SAMPLING_ALPHA}"
  echo "IMAGE_SIZE=${IMAGE_SIZE} BATCH_SIZE=${BATCH_SIZE} LOSS=${LOSS} LR=${LR} EPOCHS=${EPOCHS}"
  echo "FORCE=${FORCE} FORCE_INCOMPLETE=${FORCE_INCOMPLETE}"

  if [ "$FORCE" = "1" ]; then
    echo "FORCE=1: removing prior checkpoint folder and metrics."
    rm -rf -- "$output_dir"
    rm -f -- "$metrics_json"
  elif history_complete "$history_json" "$EPOCHS" && [ -f "$metrics_json" ]; then
    echo "Complete run and metrics found; skipping."
    rebuild_summary
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
      rebuild_summary
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
      --augment-mode "$AUGMENT_MODE" \
      --augment-prob "$AUGMENT_PROB" \
      --damage-augment-threshold "$DAMAGE_AUGMENT_THRESHOLD" \
      --sampler "$SAMPLER" \
      --damage-sampling-alpha "$DAMAGE_SAMPLING_ALPHA" \
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

  rebuild_summary
}

ensure_split
name="$(experiment_name)"
run_log="${HOME}/scratch/CrisisMap-AI/run_logs/${name}-${JOB_ID}.log"
run_job 2>&1 | tee "$run_log"
