#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/building_v2_sweep.csv}"
RUNNER="${RUNNER:-slurm/run_building_v2_config.sh}"

if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: Missing config CSV: $CONFIG" >&2
  exit 2
fi
if [[ ! -f "$RUNNER" ]]; then
  echo "ERROR: Missing runner: $RUNNER" >&2
  exit 2
fi

echo "Submitting Building v2 sweep from ${CONFIG}"
echo "Runner: ${RUNNER}"
echo "Jobs are independent; no dependency chain is used."

submitted=0

while IFS=, read -r \
  experiment \
  model \
  train_csv \
  val_csv \
  test_csv \
  input_mode \
  train_mode \
  crop_size \
  loss \
  augment_mode \
  sampler \
  sampler_alpha \
  rare_building_crop_prob \
  rare_building_crop_alpha \
  boundary_loss_weight \
  lr \
  image_size \
  batch_size \
  epochs \
  time_limit \
  num_workers
do
  if [[ -z "${experiment}" || "${experiment}" == "experiment" ]]; then
    continue
  fi

  job_name="b2_${experiment}"
  job_name="${job_name:0:48}"

  export_arg="ALL"
  export_arg+=",EXPERIMENT=${experiment}"
  export_arg+=",MODEL=${model}"
  export_arg+=",TRAIN_CSV=${train_csv}"
  export_arg+=",VAL_CSV=${val_csv}"
  export_arg+=",TEST_CSV=${test_csv}"
  export_arg+=",INPUT_MODE=${input_mode}"
  export_arg+=",TRAIN_MODE=${train_mode}"
  export_arg+=",CROP_SIZE=${crop_size}"
  export_arg+=",LOSS=${loss}"
  export_arg+=",AUGMENT_MODE=${augment_mode}"
  export_arg+=",SAMPLER=${sampler}"
  export_arg+=",SAMPLER_ALPHA=${sampler_alpha}"
  export_arg+=",RARE_BUILDING_CROP_PROB=${rare_building_crop_prob}"
  export_arg+=",RARE_BUILDING_CROP_ALPHA=${rare_building_crop_alpha}"
  export_arg+=",BOUNDARY_LOSS_WEIGHT=${boundary_loss_weight}"
  export_arg+=",LR=${lr}"
  export_arg+=",IMAGE_SIZE=${image_size}"
  export_arg+=",BATCH_SIZE=${batch_size}"
  export_arg+=",EPOCHS=${epochs}"
  export_arg+=",NUM_WORKERS=${num_workers}"

  job_id=$(sbatch \
    --parsable \
    --job-name="${job_name}" \
    --time="${time_limit}" \
    --export="${export_arg}" \
    "$RUNNER")

  echo "${job_id} ${experiment} (${model}, ${train_mode}, ${loss}, time=${time_limit})"
  submitted=$((submitted + 1))
done < "$CONFIG"

echo "Submitted ${submitted} Building v2 jobs."
