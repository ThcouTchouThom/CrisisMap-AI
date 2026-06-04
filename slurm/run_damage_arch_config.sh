#!/usr/bin/env bash
#SBATCH --job-name=damage_arch
#SBATCH --account=def-zonata_gpu
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --output=/home/tgrjlt2/scratch/CrisisMap-AI/logs/%x-%j.out
#SBATCH --error=/home/tgrjlt2/scratch/CrisisMap-AI/logs/%x-%j.err
#SBATCH --mail-user=t.gourjault@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

# Email notifications to avoid frequent scheduler polling.
# Generic one-row runner for Axis 2 damage architecture experiments.

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

ROOT="${ROOT:-data/raw/xbd/train}"
FORCE="${FORCE:-0}"
FORCE_INCOMPLETE="${FORCE_INCOMPLETE:-0}"
RESUME_INCOMPLETE="${RESUME_INCOMPLETE:-0}"
JOB_ID="${SLURM_JOB_ID:-manual}"

required_vars=(
  EXPERIMENT MODEL SPLIT TRAIN_CSV VAL_CSV TEST_CSV IMAGE_SIZE BATCH_SIZE EPOCHS
  LOSS CLASS_WEIGHTS LR AUGMENT_MODE AUGMENT_PROB DAMAGE_AUGMENT_THRESHOLD
  SAMPLER DAMAGE_SAMPLING_ALPHA HIGH_DAMAGE_THRESHOLD BASE_CHANNELS NUM_WORKERS
)
for name in "${required_vars[@]}"; do
  if [ -z "${!name:-}" ]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 1
  fi
done

CLASS_WEIGHTS="${CLASS_WEIGHTS//;/ }"
OUTPUT_DIR="outputs/checkpoints/${EXPERIMENT}"
HISTORY_JSON="${OUTPUT_DIR}/metrics_history.json"
BEST_CHECKPOINT="${OUTPUT_DIR}/best_damage_arch.pt"
LAST_CHECKPOINT="${OUTPUT_DIR}/last_damage_arch.pt"
METRICS_JSON="outputs/predictions/${EXPERIMENT}_test_metrics.json"
METRICS_CSV="outputs/predictions/${EXPERIMENT}_test_metrics.csv"
RUN_LOG="${HOME}/scratch/CrisisMap-AI/run_logs/${EXPERIMENT}-${JOB_ID}.log"

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

history_epoch_count() {
  local history_json="$1"
  [ -f "$history_json" ] || { echo "0"; return 0; }
  python - "$history_json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    print("0")
    sys.exit(0)
history = data if isinstance(data, list) else data.get("history") if isinstance(data, dict) else None
print(len(history) if isinstance(history, list) else 0)
PY
}

is_stale_empty_or_missing() {
  if [ ! -d "$OUTPUT_DIR" ]; then
    return 0
  fi
  [ ! -f "$HISTORY_JSON" ] && \
    [ ! -f "$BEST_CHECKPOINT" ] && \
    [ ! -f "$LAST_CHECKPOINT" ] && \
    [ ! -f "$METRICS_JSON" ] && \
    [ ! -f "$METRICS_CSV" ]
}

ensure_inputs() {
  local missing=0
  for path in "$TRAIN_CSV" "$VAL_CSV" "$TEST_CSV"; do
    if [ ! -f "$path" ]; then
      echo "Missing split CSV: $path" >&2
      missing=1
    fi
  done
  if [ "$missing" = "1" ]; then
    exit 1
  fi
}

run_job() {
  local weights_array
  read -r -a weights_array <<< "$CLASS_WEIGHTS"

  echo "Axis 2 damage architecture job"
  echo "Experiment: ${EXPERIMENT}"
  echo "Model: ${MODEL}"
  echo "Split: ${SPLIT}"
  echo "Train: ${TRAIN_CSV}"
  echo "Val: ${VAL_CSV}"
  echo "Test: ${TEST_CSV}"
  echo "IMAGE_SIZE=${IMAGE_SIZE} BATCH_SIZE=${BATCH_SIZE} EPOCHS=${EPOCHS}"
  echo "LOSS=${LOSS} CLASS_WEIGHTS=${CLASS_WEIGHTS} LR=${LR}"
  echo "AUGMENT_MODE=${AUGMENT_MODE} SAMPLER=${SAMPLER} ALPHA=${DAMAGE_SAMPLING_ALPHA}"
  drop_last_args=()
  if [[ "$MODEL" == *deeplabv3plus* ]]; then
    drop_last_args=(--drop-last-train)
    echo "DeepLabV3+ detected: enabling --drop-last-train for train loader BatchNorm safety."
  else
    echo "Drop last train batch: false"
  fi
  echo "Output dir: ${OUTPUT_DIR}"
  echo "History exists: $([ -f "$HISTORY_JSON" ] && echo yes || echo no)"
  echo "History epoch count: $(history_epoch_count "$HISTORY_JSON")"
  echo "Best checkpoint exists: $([ -f "$BEST_CHECKPOINT" ] && echo yes || echo no)"
  echo "Last checkpoint exists: $([ -f "$LAST_CHECKPOINT" ] && echo yes || echo no)"
  echo "Metrics JSON exists: $([ -f "$METRICS_JSON" ] && echo yes || echo no)"
  echo "Metrics CSV exists: $([ -f "$METRICS_CSV" ] && echo yes || echo no)"
  echo "FORCE=${FORCE} FORCE_INCOMPLETE=${FORCE_INCOMPLETE} RESUME_INCOMPLETE=${RESUME_INCOMPLETE}"

  if [ "$FORCE" = "1" ]; then
    echo "Selected action: train_fresh_force"
    rm -rf -- "$OUTPUT_DIR"
    rm -f -- "$METRICS_JSON" "$METRICS_CSV"
  elif history_complete "$HISTORY_JSON" "$EPOCHS" && [ -f "$METRICS_JSON" ] && [ -f "$METRICS_CSV" ]; then
    echo "Selected action: skip_complete"
    return 0
  elif history_complete "$HISTORY_JSON" "$EPOCHS" && [ -f "$BEST_CHECKPOINT" ]; then
    echo "Selected action: evaluate_only"
  elif is_stale_empty_or_missing; then
    echo "Selected action: train_fresh_stale_or_missing"
    rm -rf -- "$OUTPUT_DIR"
    rm -f -- "$METRICS_JSON" "$METRICS_CSV"
  elif [ "$RESUME_INCOMPLETE" = "1" ] && [ -f "$LAST_CHECKPOINT" ]; then
    echo "Selected action: resume"
    rm -f -- "$METRICS_JSON" "$METRICS_CSV"
  elif [ "$FORCE_INCOMPLETE" = "1" ]; then
    echo "Selected action: train_fresh_force_incomplete"
    rm -rf -- "$OUTPUT_DIR"
    rm -f -- "$METRICS_JSON" "$METRICS_CSV"
  else
    echo "Selected action: fail_incomplete"
    echo "ERROR: Incomplete output exists and will not be evaluated." >&2
    echo "Use RESUME_INCOMPLETE=1 if last_damage_arch.pt exists, or FORCE_INCOMPLETE=1 to retrain." >&2
    exit 1
  fi

  if ! history_complete "$HISTORY_JSON" "$EPOCHS"; then
    resume_args=()
    if [ "$RESUME_INCOMPLETE" = "1" ] && [ -f "$LAST_CHECKPOINT" ]; then
      resume_args=(--resume-checkpoint "$LAST_CHECKPOINT")
    fi

    python -u scripts/train_damage_architecture.py \
      --root "$ROOT" \
      --train-csv "$TRAIN_CSV" \
      --val-csv "$VAL_CSV" \
      --output-dir "$OUTPUT_DIR" \
      --model "$MODEL" \
      --image-size "$IMAGE_SIZE" \
      --batch-size "$BATCH_SIZE" \
      --epochs "$EPOCHS" \
      --target-mode 3-class \
      --loss "$LOSS" \
      --class-weights "${weights_array[@]}" \
      --lr "$LR" \
      --num-workers "$NUM_WORKERS" \
      --augment-mode "$AUGMENT_MODE" \
      --augment-prob "$AUGMENT_PROB" \
      --damage-augment-threshold "$DAMAGE_AUGMENT_THRESHOLD" \
      --sampler "$SAMPLER" \
      --damage-sampling-alpha "$DAMAGE_SAMPLING_ALPHA" \
      --high-damage-threshold "$HIGH_DAMAGE_THRESHOLD" \
      --base-channels "$BASE_CHANNELS" \
      --amp \
      "${drop_last_args[@]}" \
      "${resume_args[@]}"
  fi

  if ! history_complete "$HISTORY_JSON" "$EPOCHS" || [ ! -f "$BEST_CHECKPOINT" ]; then
    echo "ERROR: Training is not complete; refusing to evaluate partial checkpoint." >&2
    exit 1
  fi

  if [ ! -f "$METRICS_JSON" ] || [ ! -f "$METRICS_CSV" ]; then
    python -u scripts/evaluate_damage_architecture.py \
      --root "$ROOT" \
      --split-csv "$TEST_CSV" \
      --checkpoint "$BEST_CHECKPOINT" \
      --output-json "$METRICS_JSON" \
      --output-csv "$METRICS_CSV" \
      --model "$MODEL" \
      --image-size "$IMAGE_SIZE" \
      --batch-size 1 \
      --target-mode 3-class \
      --num-workers "$NUM_WORKERS" \
      --base-channels "$BASE_CHANNELS" \
      --amp
  fi
}

ensure_inputs
run_job 2>&1 | tee "$RUN_LOG"
