#!/usr/bin/env bash
#SBATCH --job-name=building_long
#SBATCH --account=def-zonata_gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=19:00:00
#SBATCH --output=/scratch/tgrjlt2/CrisisMap-AI/logs/%x-%j.out
#SBATCH --error=/scratch/tgrjlt2/CrisisMap-AI/logs/%x-%j.err
#SBATCH --mail-user=t.gourjault@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

# Email notifications to avoid frequent scheduler polling.
# Generic one-row runner for the long building segmentation campaign.

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
export TRITON_CACHE_DIR="${SCRATCH}/CrisisMap-AI/triton_cache"
mkdir -p "${TRITON_CACHE_DIR}" "${HOME}/scratch/CrisisMap-AI/run_logs"
mkdir -p outputs/checkpoints outputs/predictions outputs/figures/building_long

ROOT="data/raw/xbd/train"
CONFIG_CSV="${CONFIG_CSV:-configs/building_long_sweep_v1.csv}"
SUMMARY_CSV="outputs/predictions/building_long_sweep_v1_summary.csv"
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
EXAMPLES_DIR="outputs/figures/building_long/${EXPERIMENT}"
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

history_status() {
  local history_json="$1"
  local expected_epochs="$2"
  if [ ! -f "$history_json" ]; then
    echo "missing,0,0"
    return 0
  fi
  python - "$history_json" "$expected_epochs" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = int(sys.argv[2])
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    print("unreadable,0,0")
    sys.exit(0)
history = data if isinstance(data, list) else data.get("history") if isinstance(data, dict) else None
if not isinstance(history, list):
    print("invalid,0,0")
    sys.exit(0)
epochs = []
for item in history:
    if isinstance(item, dict) and "epoch" in item:
        try:
            epochs.append(int(item["epoch"]))
        except (TypeError, ValueError):
            pass
last_epoch = max(epochs) if epochs else 0
state = "complete" if len(history) >= expected and last_epoch >= expected else "incomplete"
print(f"{state},{len(history)},{last_epoch}")
PY
}

file_bool() {
  local path="$1"
  if [ -f "$path" ]; then
    echo "yes"
  else
    echo "no"
  fi
}

rebuild_summary() {
  python scripts/rebuild_building_long_summary.py \
    --config "$CONFIG_CSV" \
    --output "$SUMMARY_CSV"
}

run_job() {
  echo "Long building segmentation job"
  echo "Experiment: ${EXPERIMENT}"
  echo "Model=${MODEL} input=${INPUT_MODE} loss=${LOSS} augment=${AUGMENT_MODE}"
  echo "Sampler=${SAMPLER} alpha=${SAMPLER_ALPHA} lr=${LR}"
  echo "Train=${TRAIN_CSV}"
  echo "Val=${VAL_CSV}"
  echo "Test=${TEST_CSV}"
  echo "IMAGE_SIZE=${IMAGE_SIZE} BATCH_SIZE=${BATCH_SIZE} EPOCHS=${EPOCHS}"
  echo "FORCE=${FORCE} FORCE_INCOMPLETE=${FORCE_INCOMPLETE} RESUME_INCOMPLETE=${RESUME_INCOMPLETE}"

  drop_last_args=()
  if [[ "$MODEL" == deeplabv3plus* ]]; then
    drop_last_args=(--drop-last-train)
    echo "DeepLabV3+ detected: enabling --drop-last-train for train loader BatchNorm safety."
  else
    echo "Drop last train batch: false"
  fi

  IFS=, read -r history_state epoch_count last_epoch <<< "$(history_status "$HISTORY_JSON" "$EPOCHS")"
  metrics_complete=0
  if [ -f "$METRICS_JSON" ] && [ -f "$METRICS_CSV" ]; then
    metrics_complete=1
  fi
  has_any_artifact=0
  if [ -f "$HISTORY_JSON" ] || [ -f "$BEST_CHECKPOINT" ] || [ -f "$LAST_CHECKPOINT" ] || [ -f "$METRICS_JSON" ] || [ -f "$METRICS_CSV" ]; then
    has_any_artifact=1
  fi

  echo "Output dir: ${OUTPUT_DIR}"
  echo "metrics_history exists: $(file_bool "$HISTORY_JSON")"
  echo "history status: ${history_state}"
  echo "epoch count: ${epoch_count}"
  echo "last epoch: ${last_epoch}"
  echo "best_building.pt exists: $(file_bool "$BEST_CHECKPOINT")"
  echo "last_building.pt exists: $(file_bool "$LAST_CHECKPOINT")"
  echo "test metrics JSON exists: $(file_bool "$METRICS_JSON")"
  echo "test metrics CSV exists: $(file_bool "$METRICS_CSV")"

  selected_action="train_fresh"
  if [ "$FORCE" = "1" ]; then
    selected_action="train_fresh"
    echo "Selected action: ${selected_action} (FORCE=1)"
    rm -rf -- "$OUTPUT_DIR" "$EXAMPLES_DIR"
    rm -f -- "$METRICS_JSON" "$METRICS_CSV"
  elif [ "$history_state" = "complete" ] && [ "$metrics_complete" = "1" ]; then
    selected_action="skip"
    echo "Selected action: ${selected_action}"
    rebuild_summary
    return 0
  elif [ "$history_state" = "complete" ] && [ -f "$BEST_CHECKPOINT" ]; then
    selected_action="evaluate_only"
    echo "Selected action: ${selected_action}"
  elif [ -d "$OUTPUT_DIR" ]; then
    if [ "$has_any_artifact" = "0" ]; then
      selected_action="train_fresh"
      echo "Selected action: ${selected_action} (stale empty output folder)"
      rm -rf -- "$OUTPUT_DIR" "$EXAMPLES_DIR"
      rm -f -- "$METRICS_JSON" "$METRICS_CSV"
    elif [ "$RESUME_INCOMPLETE" = "1" ] && [ -f "$LAST_CHECKPOINT" ]; then
      selected_action="resume"
      echo "Selected action: ${selected_action}"
      rm -f -- "$METRICS_JSON" "$METRICS_CSV"
    elif [ "$FORCE_INCOMPLETE" = "1" ]; then
      selected_action="train_fresh"
      echo "Selected action: ${selected_action} (FORCE_INCOMPLETE=1)"
      rm -rf -- "$OUTPUT_DIR" "$EXAMPLES_DIR"
      rm -f -- "$METRICS_JSON" "$METRICS_CSV"
    else
      selected_action="fail_incomplete"
      echo "Selected action: ${selected_action}"
      echo "ERROR: Incomplete output exists and will not be evaluated." >&2
      echo "Use RESUME_INCOMPLETE=1 if last_building.pt exists, or FORCE_INCOMPLETE=1 to retrain." >&2
      rebuild_summary
      return 1
    fi
  else
    selected_action="train_fresh"
    echo "Selected action: ${selected_action} (missing output folder)"
    rm -f -- "$METRICS_JSON" "$METRICS_CSV"
  fi

  if [ "$selected_action" != "evaluate_only" ] && ! history_complete "$HISTORY_JSON" "$EPOCHS"; then
    resume_args=()
    if [ "$selected_action" = "resume" ] && [ -f "$LAST_CHECKPOINT" ]; then
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
      "${drop_last_args[@]}" \
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
