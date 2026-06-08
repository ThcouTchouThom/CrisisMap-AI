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

IFS=, read -r -a header < "${CONFIG}"
has_damage_model=0
for column in "${header[@]}"; do
  if [[ "${column}" == "damage_model" ]]; then
    has_damage_model=1
  fi
done

while IFS=, read -r -a fields
do
  if [[ "${#fields[@]}" -eq 0 ]]; then
    continue
  fi

  experiment="${fields[0]:-}"
  if [[ -z "${experiment}" || "${experiment}" == "experiment" ]]; then
    continue
  fi

  task="${fields[1]:-}"
  damage_checkpoint="${fields[2]:-}"
  if [[ "${has_damage_model}" == "1" ]]; then
    damage_model="${fields[3]:-unet}"
    damage_tta="${fields[4]:-}"
    building_models="${fields[5]:-}"
    building_checkpoints="${fields[6]:-}"
    building_input_modes="${fields[7]:-}"
    building_tta="${fields[8]:-}"
    ensemble_modes="${fields[9]:-}"
    thresholds="${fields[10]:-}"
    postprocess_modes="${fields[11]:-}"
    split_csv="${fields[12]:-}"
    image_size="${fields[13]:-}"
    batch_size="${fields[14]:-}"
    num_workers="${fields[15]:-}"
    time_limit="${fields[16]:-}"
  else
    damage_model="${DAMAGE_MODEL:-unet}"
    damage_tta="${fields[3]:-}"
    building_models="${fields[4]:-}"
    building_checkpoints="${fields[5]:-}"
    building_input_modes="${fields[6]:-}"
    building_tta="${fields[7]:-}"
    ensemble_modes="${fields[8]:-}"
    thresholds="${fields[9]:-}"
    postprocess_modes="${fields[10]:-}"
    split_csv="${fields[11]:-}"
    image_size="${fields[12]:-}"
    batch_size="${fields[13]:-}"
    num_workers="${fields[14]:-}"
    time_limit="${fields[15]:-}"
  fi
  time_limit="${time_limit//$'\r'/}"

  job_name="bp_${experiment}"
  job_name="${job_name:0:48}"

  export_arg="ALL"
  export_arg+=",EXPERIMENT=${experiment}"
  export_arg+=",TASK=${task}"
  export_arg+=",DAMAGE_CHECKPOINT=${damage_checkpoint}"
  export_arg+=",DAMAGE_MODEL=${damage_model}"
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
done < <(tail -n +2 "${CONFIG}")

echo "Submitted ${submitted} building post-processing jobs."
