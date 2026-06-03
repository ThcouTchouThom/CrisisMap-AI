#!/bin/bash
set -euo pipefail

CONFIG="${1:-configs/multihead_damage_sweep_v1.csv}"
RUNNER="${RUNNER:-slurm/run_multihead_damage_config.sh}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: Missing config CSV: ${CONFIG}" >&2
  exit 2
fi
if [[ ! -f "${RUNNER}" ]]; then
  echo "ERROR: Missing runner: ${RUNNER}" >&2
  exit 2
fi

echo "Submitting Axis 3 multi-head damage sweep from ${CONFIG}"
echo "Runner: ${RUNNER}"
echo "Jobs are independent; disabled/planned rows are skipped."

submitted=0
skipped=0

while IFS=, read -r \
  enabled \
  experiment \
  model \
  target_mode \
  label_mode \
  train_mode \
  image_size \
  crop_size \
  train_csv \
  val_csv \
  test_csv \
  building_loss \
  damage_loss \
  lambda_building \
  lambda_damage \
  lr \
  weight_decay \
  batch_size \
  epochs \
  time_limit \
  num_workers \
  augment \
  rare_damage_crop_prob \
  rare_damage_crop_alpha \
  status
do
  if [[ -z "${experiment}" || "${experiment}" == "experiment" ]]; then
    continue
  fi
  if [[ "${enabled}" != "1" && "${enabled}" != "true" && "${enabled}" != "TRUE" ]]; then
    echo "Skipping disabled/planned row: ${experiment} (${status})"
    skipped=$((skipped + 1))
    continue
  fi

  job_name="mh_${experiment}"
  job_name="${job_name:0:48}"

  export_arg="ALL"
  export_arg+=",ENABLED=${enabled}"
  export_arg+=",EXPERIMENT=${experiment}"
  export_arg+=",MODEL=${model}"
  export_arg+=",TARGET_MODE=${target_mode}"
  export_arg+=",LABEL_MODE=${label_mode}"
  export_arg+=",TRAIN_MODE=${train_mode}"
  export_arg+=",IMAGE_SIZE=${image_size}"
  export_arg+=",CROP_SIZE=${crop_size}"
  export_arg+=",TRAIN_CSV=${train_csv}"
  export_arg+=",VAL_CSV=${val_csv}"
  export_arg+=",TEST_CSV=${test_csv}"
  export_arg+=",BUILDING_LOSS=${building_loss}"
  export_arg+=",DAMAGE_LOSS=${damage_loss}"
  export_arg+=",LAMBDA_BUILDING=${lambda_building}"
  export_arg+=",LAMBDA_DAMAGE=${lambda_damage}"
  export_arg+=",LR=${lr}"
  export_arg+=",WEIGHT_DECAY=${weight_decay}"
  export_arg+=",BATCH_SIZE=${batch_size}"
  export_arg+=",EPOCHS=${epochs}"
  export_arg+=",NUM_WORKERS=${num_workers}"
  export_arg+=",AUGMENT=${augment}"
  export_arg+=",RARE_DAMAGE_CROP_PROB=${rare_damage_crop_prob}"
  export_arg+=",RARE_DAMAGE_CROP_ALPHA=${rare_damage_crop_alpha}"

  job_id=$(sbatch \
    --parsable \
    --job-name="${job_name}" \
    --time="${time_limit}" \
    --export="${export_arg}" \
    "${RUNNER}")

  echo "${job_id} ${experiment} (${model}, ${label_mode}, ${damage_loss}, time=${time_limit})"
  submitted=$((submitted + 1))
done < "${CONFIG}"

echo "Submitted ${submitted} Axis 3 multi-head jobs. Skipped ${skipped} disabled/planned rows."
