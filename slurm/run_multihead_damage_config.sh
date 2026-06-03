#!/bin/bash
#SBATCH --job-name=mh_damage
#SBATCH --account=def-zonata_gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=${SCRATCH}/CrisisMap-AI/logs/%x-%j.out
#SBATCH --error=${SCRATCH}/CrisisMap-AI/logs/%x-%j.err
# Email notifications to avoid frequent scheduler polling
#SBATCH --mail-user=t.gourjault@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

set -euo pipefail

required_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "ERROR: Required environment variable ${name} is not set." >&2
    exit 2
  fi
}

required_env ENABLED
required_env EXPERIMENT
required_env MODEL
required_env TARGET_MODE
required_env LABEL_MODE
required_env TRAIN_MODE
required_env IMAGE_SIZE
required_env CROP_SIZE
required_env TRAIN_CSV
required_env VAL_CSV
required_env TEST_CSV
required_env BUILDING_LOSS
required_env DAMAGE_LOSS
required_env LAMBDA_BUILDING
required_env LAMBDA_DAMAGE
required_env LR
required_env WEIGHT_DECAY
required_env BATCH_SIZE
required_env EPOCHS
required_env NUM_WORKERS
required_env AUGMENT
required_env RARE_DAMAGE_CROP_PROB
required_env RARE_DAMAGE_CROP_ALPHA

if [[ "${ENABLED}" != "1" && "${ENABLED}" != "true" && "${ENABLED}" != "TRUE" ]]; then
  echo "Experiment ${EXPERIMENT} is disabled/planned in the config; skipping."
  exit 0
fi

if [[ -z "${SCRATCH:-}" ]]; then
  echo "ERROR: SCRATCH is not set. This runner expects Rorqual scratch paths." >&2
  exit 2
fi

CODE_DIR="${CODE_DIR:-${HOME}/work/CrisisMap-AI}"
VENV_PATH="${VENV_PATH:-${HOME}/virtualenvs/crisismap-ai/bin/activate}"
DATA_ROOT="${DATA_ROOT:-data/raw/xbd/train}"
FORCE="${FORCE:-0}"
FORCE_INCOMPLETE="${FORCE_INCOMPLETE:-0}"
RESUME_INCOMPLETE="${RESUME_INCOMPLETE:-0}"

export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning}"
export TRITON_CACHE_DIR="${SCRATCH}/CrisisMap-AI/triton_cache"
RUN_LOG_DIR="${SCRATCH}/CrisisMap-AI/run_logs"
SLURM_LOG_DIR="${SCRATCH}/CrisisMap-AI/logs"
mkdir -p "${TRITON_CACHE_DIR}" "${RUN_LOG_DIR}" "${SLURM_LOG_DIR}"

cd "${CODE_DIR}"

module load StdEnv/2023
module load python/3.11
module load gcc
module load arrow/23.0.1
module load cuda
module load opencv/4.13.0

source "${VENV_PATH}"

if [[ "${TARGET_MODE}" == "5-class" || "${LABEL_MODE}" == "5-class" ]]; then
  echo "Running automatic 5-class label audit before enabling ${EXPERIMENT}."
  python - "${DATA_ROOT}" "${TRAIN_CSV}" "${VAL_CSV}" "${TEST_CSV}" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
csv_paths = [Path(value) for value in sys.argv[2:]]
observed = set()

for csv_path in csv_paths:
    df = pd.read_csv(csv_path)
    if "target_value_counts" in df.columns:
        for value in df["target_value_counts"].dropna():
            try:
                counts = json.loads(str(value))
            except json.JSONDecodeError:
                continue
            observed.update(int(float(key)) for key in counts.keys())
        continue

    from crisismap.data.xbd_dataset import load_target_mask, resolve_data_path

    if "target" not in df.columns:
        raise SystemExit(f"{csv_path} has no target column for 5-class audit.")
    for _, row in df.iterrows():
        mask = load_target_mask(resolve_data_path(root, row["target"]), image_size=1024)
        observed.update(int(v) for v in pd.unique(mask.reshape(-1)))

required = {2, 3, 4}
if not required.issubset(observed):
    raise SystemExit(
        "5-class audit failed: expected separate minor/major/destroyed labels "
        f"{sorted(required)}, observed labels {sorted(observed)}. "
        "Keep this row disabled/planned until original 5-class labels are available."
    )
print(f"5-class label audit OK. Observed labels: {sorted(observed)}")
PY
fi

CHECKPOINT_DIR="outputs/checkpoints/${EXPERIMENT}"
PRED_DIR="outputs/predictions/multihead_damage"
METRICS_JSON="${PRED_DIR}/${EXPERIMENT}_test_metrics.json"
METRICS_CSV="${PRED_DIR}/${EXPERIMENT}_test_metrics.csv"
DOWNSTREAM_JSON="${PRED_DIR}/${EXPERIMENT}_downstream_metrics.json"
DOWNSTREAM_CSV="${PRED_DIR}/${EXPERIMENT}_downstream_metrics.csv"
HISTORY_JSON="${CHECKPOINT_DIR}/metrics_history.json"
BEST_CHECKPOINT="${CHECKPOINT_DIR}/best_multihead_damage.pt"
LAST_CHECKPOINT="${CHECKPOINT_DIR}/last_multihead_damage.pt"
RUN_LOG="${RUN_LOG_DIR}/${EXPERIMENT}-${SLURM_JOB_ID:-manual}.log"

mkdir -p "${PRED_DIR}"

epoch_count() {
  local history_path="$1"
  if [[ ! -f "${history_path}" ]]; then
    echo 0
    return
  fi
  python -c "import json,sys; p=sys.argv[1]; data=json.load(open(p, encoding='utf-8')); print(len(data) if isinstance(data, list) else 0)" "${history_path}"
}

EPOCH_COUNT="$(epoch_count "${HISTORY_JSON}")"
COMPLETE=0
if [[ "${EPOCH_COUNT}" -ge "${EPOCHS}" && -f "${METRICS_JSON}" && -f "${DOWNSTREAM_JSON}" && -f "${BEST_CHECKPOINT}" ]]; then
  COMPLETE=1
fi

echo "Experiment: ${EXPERIMENT}"
echo "Model: ${MODEL}"
echo "Target mode: ${TARGET_MODE}"
echo "Label mode: ${LABEL_MODE}"
echo "Train mode: ${TRAIN_MODE}"
echo "Building loss: ${BUILDING_LOSS}"
echo "Damage loss: ${DAMAGE_LOSS}"
echo "Checkpoint dir: ${CHECKPOINT_DIR}"
echo "History epochs: ${EPOCH_COUNT}/${EPOCHS}"
echo "Best checkpoint exists: $([[ -f "${BEST_CHECKPOINT}" ]] && echo yes || echo no)"
echo "Last checkpoint exists: $([[ -f "${LAST_CHECKPOINT}" ]] && echo yes || echo no)"
echo "Metrics JSON exists: $([[ -f "${METRICS_JSON}" ]] && echo yes || echo no)"
echo "Downstream JSON exists: $([[ -f "${DOWNSTREAM_JSON}" ]] && echo yes || echo no)"
echo "FORCE=${FORCE} FORCE_INCOMPLETE=${FORCE_INCOMPLETE} RESUME_INCOMPLETE=${RESUME_INCOMPLETE}"

evaluate_complete_checkpoint() {
  echo "Action: evaluate checkpoint"
  python -u scripts/evaluate_multihead_damage.py \
    --checkpoint "${BEST_CHECKPOINT}" \
    --root "${DATA_ROOT}" \
    --split-csv "${TEST_CSV}" \
    --output-json "${METRICS_JSON}" \
    --output-csv "${METRICS_CSV}" \
    --model "${MODEL}" \
    --target-mode "${TARGET_MODE}" \
    --label-mode "${LABEL_MODE}" \
    --image-size "${IMAGE_SIZE}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --device cuda \
    --amp 2>&1 | tee -a "${RUN_LOG}"
  python -u scripts/evaluate_multihead_downstream.py \
    --checkpoint "${BEST_CHECKPOINT}" \
    --root "${DATA_ROOT}" \
    --split-csv "${TEST_CSV}" \
    --output-json "${DOWNSTREAM_JSON}" \
    --output-csv "${DOWNSTREAM_CSV}" \
    --model "${MODEL}" \
    --target-mode "${TARGET_MODE}" \
    --label-mode "${LABEL_MODE}" \
    --image-size "${IMAGE_SIZE}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --device cuda \
    --amp 2>&1 | tee -a "${RUN_LOG}"
}

if [[ "${COMPLETE}" == "1" && "${FORCE}" != "1" ]]; then
  echo "Action: skip complete run"
  exit 0
fi

if [[ "${EPOCH_COUNT}" -ge "${EPOCHS}" && -f "${BEST_CHECKPOINT}" && ( ! -f "${METRICS_JSON}" || ! -f "${DOWNSTREAM_JSON}" ) ]]; then
  echo "Action: evaluate_only"
  evaluate_complete_checkpoint
  exit 0
fi

if [[ "${EPOCH_COUNT}" -ge "${EPOCHS}" && ! -f "${BEST_CHECKPOINT}" && "${FORCE}" != "1" && "${FORCE_INCOMPLETE}" != "1" ]]; then
  echo "ERROR: History has expected epochs but best checkpoint is missing." >&2
  echo "Use FORCE_INCOMPLETE=1 only to retrain this experiment from scratch." >&2
  exit 2
fi

if [[ "${FORCE}" == "1" || "${FORCE_INCOMPLETE}" == "1" ]]; then
  echo "Action: force_train_fresh"
  rm -rf "${CHECKPOINT_DIR}"
  rm -f "${METRICS_JSON}" "${METRICS_CSV}" "${DOWNSTREAM_JSON}" "${DOWNSTREAM_CSV}"
  EPOCH_COUNT=0
fi

TRAIN_EXTRA_ARGS=()
if [[ "${EPOCH_COUNT}" -gt 0 ]]; then
  if [[ "${EPOCH_COUNT}" -lt "${EPOCHS}" && "${RESUME_INCOMPLETE}" == "1" && -f "${LAST_CHECKPOINT}" ]]; then
    echo "Action: resume from ${LAST_CHECKPOINT}"
    TRAIN_EXTRA_ARGS+=(--resume-checkpoint "${LAST_CHECKPOINT}")
  else
    echo "ERROR: Incomplete checkpoint folder exists." >&2
    echo "Use RESUME_INCOMPLETE=1 with a last checkpoint or FORCE_INCOMPLETE=1 to retrain." >&2
    exit 2
  fi
else
  echo "Action: train_fresh"
fi

TRAIN_ARGS=()
if [[ "${AUGMENT}" == "1" || "${AUGMENT}" == "true" || "${AUGMENT}" == "TRUE" ]]; then
  TRAIN_ARGS+=(--augment)
fi
if [[ "${RARE_DAMAGE_CROP_ALPHA}" != "none" && -n "${RARE_DAMAGE_CROP_ALPHA}" ]]; then
  TRAIN_ARGS+=(--rare-damage-crop-alpha "${RARE_DAMAGE_CROP_ALPHA}")
fi

python -u scripts/train_multihead_damage.py \
  --root "${DATA_ROOT}" \
  --train-csv "${TRAIN_CSV}" \
  --val-csv "${VAL_CSV}" \
  --output-dir "${CHECKPOINT_DIR}" \
  --model "${MODEL}" \
  --target-mode "${TARGET_MODE}" \
  --label-mode "${LABEL_MODE}" \
  --train-mode "${TRAIN_MODE}" \
  --image-size "${IMAGE_SIZE}" \
  --crop-size "${CROP_SIZE}" \
  --rare-damage-crop-prob "${RARE_DAMAGE_CROP_PROB}" \
  --building-loss "${BUILDING_LOSS}" \
  --damage-loss "${DAMAGE_LOSS}" \
  --lambda-building "${LAMBDA_BUILDING}" \
  --lambda-damage "${LAMBDA_DAMAGE}" \
  --batch-size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --num-workers "${NUM_WORKERS}" \
  --device cuda \
  --amp \
  "${TRAIN_ARGS[@]}" \
  "${TRAIN_EXTRA_ARGS[@]}" 2>&1 | tee "${RUN_LOG}"

if [[ ! -f "${BEST_CHECKPOINT}" ]]; then
  echo "ERROR: Best checkpoint missing after training: ${BEST_CHECKPOINT}" >&2
  exit 2
fi

evaluate_complete_checkpoint

echo "Finished ${EXPERIMENT}"
echo "Metrics: ${METRICS_JSON}"
echo "Downstream metrics: ${DOWNSTREAM_JSON}"
