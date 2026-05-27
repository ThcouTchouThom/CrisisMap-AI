#!/usr/bin/env bash
#SBATCH --job-name=building100
#SBATCH --account=def-zonata_gpu
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:30:00
#SBATCH --output=/home/tgrjlt2/scratch/CrisisMap-AI/logs/%x-%j.out
#SBATCH --error=/home/tgrjlt2/scratch/CrisisMap-AI/logs/%x-%j.err
#SBATCH --mail-user=t.gourjault@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

# Email notifications to avoid frequent scheduler polling.
# Generic one-row Building100 runner. Time limit and job name are normally
# overridden by slurm/submit_building100_sweep_v1.sh from the CSV.

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
mkdir -p outputs/checkpoints outputs/predictions outputs/figures/building100

ROOT="data/raw/xbd/train"
CONFIG_CSV="${CONFIG_CSV:-configs/building100_sweep_v1.csv}"
SUMMARY_CSV="outputs/predictions/building100_sweep_v1_summary.csv"
FORCE="${FORCE:-0}"
FORCE_INCOMPLETE="${FORCE_INCOMPLETE:-0}"
RESUME_INCOMPLETE="${RESUME_INCOMPLETE:-0}"
JOB_ID="${SLURM_JOB_ID:-manual}"

required_vars=(
  EXPERIMENT MODEL TRAIN_CSV VAL_CSV TEST_CSV INPUT_MODE LOSS AUGMENT_MODE
  SAMPLER SAMPLER_ALPHA LR IMAGE_SIZE BATCH_SIZE EPOCHS NUM_WORKERS
)
for name in "${required_vars[@]}"; do
  if [ -z "${!name:-}" ]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 1
  fi
done

OUTPUT_DIR="outputs/checkpoints/${EXPERIMENT}"
HISTORY_JSON="${OUTPUT_DIR}/metrics_history.json"
BEST_CHECKPOINT="${OUTPUT_DIR}/best_building.pt"
LAST_CHECKPOINT="${OUTPUT_DIR}/last_building.pt"
METRICS_JSON="outputs/predictions/${EXPERIMENT}_building_test_metrics.json"
METRICS_CSV="outputs/predictions/${EXPERIMENT}_building_test_metrics.csv"
EXAMPLES_DIR="outputs/figures/building100/${EXPERIMENT}"
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

rebuild_summary() {
  python scripts/rebuild_building100_summary.py \
    --config "$CONFIG_CSV" \
    --output "$SUMMARY_CSV"
}

run_job() {
  echo "Building100 sweep job"
  echo "Experiment: ${EXPERIMENT}"
  echo "Model=${MODEL} input=${INPUT_MODE} loss=${LOSS} augment=${AUGMENT_MODE}"
  echo "Sampler=${SAMPLER} alpha=${SAMPLER_ALPHA} lr=${LR}"
  echo "Train=${TRAIN_CSV}"
  echo "Val=${VAL_CSV}"
  echo "Test=${TEST_CSV}"
  echo "FORCE=${FORCE} FORCE_INCOMPLETE=${FORCE_INCOMPLETE} RESUME_INCOMPLETE=${RESUME_INCOMPLETE}"

  if [ "$FORCE" = "1" ]; then
    echo "FORCE=1: removing prior outputs for this experiment."
    rm -rf -- "$OUTPUT_DIR" "$EXAMPLES_DIR"
    rm -f -- "$METRICS_JSON" "$METRICS_CSV"
  elif history_complete "$HISTORY_JSON" "$EPOCHS" && [ -f "$METRICS_JSON" ] && [ -f "$METRICS_CSV" ]; then
    echo "Complete run and metrics found; skipping."
    rebuild_summary
    return 0
  elif [ -d "$OUTPUT_DIR" ]; then
    if history_complete "$HISTORY_JSON" "$EPOCHS" && [ -f "$BEST_CHECKPOINT" ]; then
      echo "Training complete; evaluating missing metrics."
    elif [ "$RESUME_INCOMPLETE" = "1" ] && [ -f "$LAST_CHECKPOINT" ]; then
      echo "RESUME_INCOMPLETE=1: resuming from last_building.pt."
      rm -f -- "$METRICS_JSON" "$METRICS_CSV"
    elif [ "$FORCE_INCOMPLETE" = "1" ]; then
      echo "FORCE_INCOMPLETE=1: removing incomplete output before retraining."
      rm -rf -- "$OUTPUT_DIR" "$EXAMPLES_DIR"
      rm -f -- "$METRICS_JSON" "$METRICS_CSV"
    else
      echo "WARNING: Incomplete output folder; not evaluating partial checkpoint."
      echo "WARNING: Relaunch with FORCE_INCOMPLETE=1 or RESUME_INCOMPLETE=1."
      rebuild_summary
      return 1
    fi
  fi

  if ! history_complete "$HISTORY_JSON" "$EPOCHS"; then
    resume_args=()
    if [ "$RESUME_INCOMPLETE" = "1" ] && [ -f "$LAST_CHECKPOINT" ]; then
      resume_args=(--resume-checkpoint "$LAST_CHECKPOINT")
    fi

    python -u scripts/train_building_segmentation.py \
      --root "$ROOT" \
      --train-csv "$TRAIN_CSV" \
      --val-csv "$VAL_CSV" \
      --output-dir "$OUTPUT_DIR" \
      --model "$MODEL" \
      --input-mode "$INPUT_MODE" \
      --image-size "$IMAGE_SIZE" \
      --batch-size "$BATCH_SIZE" \
      --epochs "$EPOCHS" \
      --lr "$LR" \
      --loss "$LOSS" \
      --num-workers "$NUM_WORKERS" \
      --device auto \
      --amp \
      --augment-mode "$AUGMENT_MODE" \
      --sampler "$SAMPLER" \
      --sampler-alpha "$SAMPLER_ALPHA" \
      --target-mode building-binary \
      "${resume_args[@]}"
  fi

  if ! history_complete "$HISTORY_JSON" "$EPOCHS" || [ ! -f "$BEST_CHECKPOINT" ]; then
    echo "ERROR: Training is not complete; refusing to evaluate partial checkpoint." >&2
    exit 1
  fi

  if [ ! -f "$METRICS_JSON" ] || [ ! -f "$METRICS_CSV" ]; then
    python -u scripts/evaluate_building_segmentation.py \
      --checkpoint "$BEST_CHECKPOINT" \
      --root "$ROOT" \
      --split-csv "$TEST_CSV" \
      --model "$MODEL" \
      --input-mode "$INPUT_MODE" \
      --image-size "$IMAGE_SIZE" \
      --target-mode building-binary \
      --batch-size 1 \
      --num-workers "$NUM_WORKERS" \
      --device auto \
      --amp \
      --thresholds 0.3 0.4 0.5 0.6 \
      --output-json "$METRICS_JSON" \
      --output-csv "$METRICS_CSV" \
      --save-examples-dir "$EXAMPLES_DIR" \
      --num-examples 4
  fi

  rebuild_summary
}

run_job 2>&1 | tee "$RUN_LOG"
