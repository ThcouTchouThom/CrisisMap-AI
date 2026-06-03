#!/bin/bash
set -euo pipefail

CONFIG="${1:-configs/building_postprocess_sweep_v1.csv}"
RUNNER="${RUNNER:-slurm/run_building_postprocess_config.sh}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: Missing config CSV: ${CONFIG}" >&2
  exit 2
fi

if [[ ! -f "${RUNNER}" ]]; then
  echo "ERROR: Missing runner: ${RUNNER}" >&2
  exit 2
fi

echo "Submitting building post-processing sweep from ${CONFIG}"
echo "Runner: ${RUNNER}"
echo "Jobs are independent; no dependency chain is used."

submitted=0

while IFS=, read -r \
  experiment \
  task \
  damage_checkpoint \
  damage_tta \
  building_models \
  building_checkpoints \
  building_input_modes \
  building_tta \
  ensemble_modes \
  thresholds \
  postprocess_modes \
  split_csv \
  image_size \
  batch_size \
  num_workers \
  time_limit
do
  if [[ -z "${experiment}" || "${experiment}" == "experiment" ]]; then
    continue
  fi

  job_name="bp_${experiment}"
  job_name="${job_name:0:48}"

  export_arg="ALL"
  export_arg+=",EXPERIMENT=${experiment}"
  export_arg+=",TASK=${task}"
  export_arg+=",DAMAGE_CHECKPOINT=${damage_checkpoint}"
  export_arg+=",DAMAGE_TTA=${damage_tta}"
  export_arg+=",BUILDING_MODELS=${building_models}"
  export_arg+=",BUILDING_CHECKPOINTS=${building_checkpoints}"
  export_arg+=",BUILDING_INPUT_MODES=${building_input_modes}"
  export_arg+=",BUILDING_TTA=${building_tta}"
  export_arg+=",ENSEMBLE_MODES=${ensemble_modes}"
  export_arg+=",THRESHOLDS=${thresholds}"
  export_arg+=",POSTPROCESS_MODES=${postprocess_modes}"
  export_arg+=",SPLIT_CSV=${split_csv}"
  export_arg+=",IMAGE_SIZE=${image_size}"
  export_arg+=",BATCH_SIZE=${batch_size}"
  export_arg+=",NUM_WORKERS=${num_workers}"

  job_id=$(sbatch \
    --parsable \
    --job-name="${job_name}" \
    --time="${time_limit}" \
    --export="${export_arg}" \
    "${RUNNER}")

  echo "${job_id} ${experiment} (${task}, time=${time_limit})"
  submitted=$((submitted + 1))
done < "${CONFIG}"

echo "Submitted ${submitted} building post-processing jobs."
