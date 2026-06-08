#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/damage_focal_tversky_v2.csv}"
RUNNER="${RUNNER:-slurm/run_damage_focal_tversky_v2_config.sh}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: Missing config CSV: ${CONFIG}" >&2
  exit 2
fi

if [[ ! -f "${RUNNER}" ]]; then
  echo "ERROR: Missing runner: ${RUNNER}" >&2
  exit 2
fi

echo "Submitting damage_focal_tversky_v2 sweep from ${CONFIG}"
echo "Runner: ${RUNNER}"
echo "Jobs are independent; no dependency chain is used."

submitted=0

while IFS=, read -r \
  experiment \
  model \
  split \
  train_csv \
  val_csv \
  test_csv \
  image_size \
  batch_size \
  epochs \
  loss \
  class_weights \
  lr \
  augment_mode \
  augment_prob \
  damage_augment_threshold \
  sampler \
  damage_sampling_alpha \
  high_damage_threshold \
  base_channels \
  seed \
  num_workers \
  time_limit
do
  if [[ -z "${experiment}" || "${experiment}" == "experiment" ]]; then
    continue
  fi
  time_limit="${time_limit//$'\r'/}"

  job_name="dftv2_${experiment}"
  job_name="${job_name:0:48}"

  export_arg="ALL"
  export_arg+=",CONFIG_CSV=${CONFIG}"
  export_arg+=",EXPERIMENT=${experiment}"
  export_arg+=",MODEL=${model}"
  export_arg+=",SPLIT=${split}"
  export_arg+=",TRAIN_CSV=${train_csv}"
  export_arg+=",VAL_CSV=${val_csv}"
  export_arg+=",TEST_CSV=${test_csv}"
  export_arg+=",IMAGE_SIZE=${image_size}"
  export_arg+=",BATCH_SIZE=${batch_size}"
  export_arg+=",EPOCHS=${epochs}"
  export_arg+=",LOSS=${loss}"
  export_arg+=",CLASS_WEIGHTS=${class_weights}"
  export_arg+=",LR=${lr}"
  export_arg+=",AUGMENT_MODE=${augment_mode}"
  export_arg+=",AUGMENT_PROB=${augment_prob}"
  export_arg+=",DAMAGE_AUGMENT_THRESHOLD=${damage_augment_threshold}"
  export_arg+=",SAMPLER=${sampler}"
  export_arg+=",DAMAGE_SAMPLING_ALPHA=${damage_sampling_alpha}"
  export_arg+=",HIGH_DAMAGE_THRESHOLD=${high_damage_threshold}"
  export_arg+=",BASE_CHANNELS=${base_channels}"
  export_arg+=",SEED=${seed}"
  export_arg+=",NUM_WORKERS=${num_workers}"

  job_id=$(sbatch \
    --parsable \
    --job-name="${job_name}" \
    --time="${time_limit}" \
    --export="${export_arg}" \
    "${RUNNER}")

  echo "${job_id} ${experiment} (epochs=${epochs}, model=${model}, time=${time_limit})"
  submitted=$((submitted + 1))
done < "${CONFIG}"

echo "Submitted ${submitted} damage_focal_tversky_v2 jobs."
