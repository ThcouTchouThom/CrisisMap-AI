#!/bin/bash
#SBATCH --job-name=building_postprocess
#SBATCH --account=def-zonata_gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/tgrjlt2/CrisisMap-AI/logs/%x-%j.out
#SBATCH --error=/scratch/tgrjlt2/CrisisMap-AI/logs/%x-%j.err
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

required_env EXPERIMENT
required_env TASK
required_env BUILDING_MODELS
required_env BUILDING_CHECKPOINTS
required_env BUILDING_INPUT_MODES
required_env BUILDING_TTA
required_env ENSEMBLE_MODES
required_env THRESHOLDS
required_env SPLIT_CSV
required_env IMAGE_SIZE
required_env BATCH_SIZE
required_env NUM_WORKERS

FORCE="${FORCE:-0}"
CODE_DIR="${CODE_DIR:-${HOME}/work/CrisisMap-AI}"
VENV_PATH="${VENV_PATH:-${HOME}/virtualenvs/crisismap-ai/bin/activate}"
DATA_ROOT="${DATA_ROOT:-data/raw/xbd/train}"

if [[ -z "${SCRATCH:-}" ]]; then
  echo "ERROR: SCRATCH is not set. This runner expects Rorqual scratch paths." >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning}"
export TRITON_CACHE_DIR="${SCRATCH}/CrisisMap-AI/triton_cache"
RUN_LOG_DIR="${SCRATCH}/CrisisMap-AI/run_logs"
SLURM_LOG_DIR="${SCRATCH}/CrisisMap-AI/logs"

mkdir -p "${TRITON_CACHE_DIR}" "${RUN_LOG_DIR}" "${SLURM_LOG_DIR}"

cd "${CODE_DIR}"
export PYTHONPATH="${CODE_DIR}/src:${CODE_DIR}:${PYTHONPATH:-}"

module load StdEnv/2023
module load python/3.11
module load gcc
module load arrow/23.0.1
module load cuda
module load opencv/4.13.0

source "${VENV_PATH}"

IFS=';' read -r -a BUILDING_MODEL_ARGS <<< "${BUILDING_MODELS}"
IFS=';' read -r -a BUILDING_CHECKPOINT_ARGS <<< "${BUILDING_CHECKPOINTS}"
IFS=';' read -r -a BUILDING_INPUT_ARGS <<< "${BUILDING_INPUT_MODES}"
IFS=';' read -r -a BUILDING_TTA_ARGS <<< "${BUILDING_TTA}"
IFS=';' read -r -a ENSEMBLE_MODE_ARGS <<< "${ENSEMBLE_MODES}"
IFS=';' read -r -a THRESHOLD_ARGS <<< "${THRESHOLDS}"

OUT_DIR="outputs/predictions/building_postprocess"
FIG_ROOT="outputs/figures/building_postprocess"
OUTPUT_JSON="${OUT_DIR}/${EXPERIMENT}.json"
OUTPUT_CSV="${OUT_DIR}/${EXPERIMENT}.csv"
EXAMPLE_DIR="${FIG_ROOT}/${EXPERIMENT}"
RUN_LOG="${RUN_LOG_DIR}/${EXPERIMENT}-${SLURM_JOB_ID:-manual}.log"

mkdir -p "${OUT_DIR}" "${FIG_ROOT}"

if [[ "${FORCE}" != "1" && -f "${OUTPUT_JSON}" && -f "${OUTPUT_CSV}" ]]; then
  echo "Complete output already exists for ${EXPERIMENT}; skipping."
  exit 0
fi

if [[ "${FORCE}" == "1" ]]; then
  echo "FORCE=1: removing previous output JSON/CSV for ${EXPERIMENT}."
  rm -f "${OUTPUT_JSON}" "${OUTPUT_CSV}"
fi

echo "Experiment: ${EXPERIMENT}"
echo "Task: ${TASK}"
echo "Split: ${SPLIT_CSV}"
echo "Building models: ${BUILDING_MODELS}"
echo "Building checkpoints: ${BUILDING_CHECKPOINTS}"
echo "Building TTA: ${BUILDING_TTA}"
echo "Ensemble modes: ${ENSEMBLE_MODES}"
echo "Thresholds: ${THRESHOLDS}"
echo "Output JSON: ${OUTPUT_JSON}"
echo "Output CSV: ${OUTPUT_CSV}"
echo "Run log: ${RUN_LOG}"

if [[ "${TASK}" == "building_eval" ]]; then
  python -u scripts/evaluate_building_tta_ensemble.py \
    --root "${DATA_ROOT}" \
    --split-csv "${SPLIT_CSV}" \
    --checkpoint "${BUILDING_CHECKPOINT_ARGS[@]}" \
    --model "${BUILDING_MODEL_ARGS[@]}" \
    --input-mode "${BUILDING_INPUT_ARGS[@]}" \
    --image-size "${IMAGE_SIZE}" \
    --target-mode building-binary \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --device auto \
    --amp \
    --tta-modes "${BUILDING_TTA_ARGS[@]}" \
    --thresholds "${THRESHOLD_ARGS[@]}" \
    --ensemble-modes "${ENSEMBLE_MODE_ARGS[@]}" \
    --component-connectivity 8 \
    --output-json "${OUTPUT_JSON}" \
    --output-csv "${OUTPUT_CSV}" \
    --save-examples-dir "${EXAMPLE_DIR}" \
    --num-examples "${NUM_EXAMPLES:-4}" 2>&1 | tee "${RUN_LOG}"
elif [[ "${TASK}" == "downstream" ]]; then
  required_env DAMAGE_CHECKPOINT
  required_env DAMAGE_TTA
  required_env POSTPROCESS_MODES
  IFS=';' read -r -a POSTPROCESS_MODE_ARGS <<< "${POSTPROCESS_MODES}"

  python -u scripts/evaluate_downstream_building_ensemble.py \
    --damage-checkpoint "${DAMAGE_CHECKPOINT}" \
    --root "${DATA_ROOT}" \
    --split-csv "${SPLIT_CSV}" \
    --image-size "${IMAGE_SIZE}" \
    --target-mode 3-class \
    --damage-model unet \
    --damage-tta "${DAMAGE_TTA}" \
    --building-checkpoint "${BUILDING_CHECKPOINT_ARGS[@]}" \
    --building-model "${BUILDING_MODEL_ARGS[@]}" \
    --building-input-mode "${BUILDING_INPUT_ARGS[@]}" \
    --building-tta "${BUILDING_TTA}" \
    --thresholds "${THRESHOLD_ARGS[@]}" \
    --ensemble-modes "${ENSEMBLE_MODE_ARGS[@]}" \
    --postprocess-modes "${POSTPROCESS_MODE_ARGS[@]}" \
    --component-connectivity 8 \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --device auto \
    --amp \
    --output-json "${OUTPUT_JSON}" \
    --output-csv "${OUTPUT_CSV}" \
    --save-examples-dir "${EXAMPLE_DIR}" \
    --num-examples "${NUM_EXAMPLES:-4}" 2>&1 | tee "${RUN_LOG}"
else
  echo "ERROR: Unsupported TASK=${TASK}. Expected building_eval or downstream." >&2
  exit 2
fi
