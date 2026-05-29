#!/usr/bin/env bash

set -euo pipefail
cd "${HOME}/work/CrisisMap-AI"

CONFIG_CSV="${CONFIG_CSV:-configs/building_long_sweep_v1.csv}"
FORCE="${FORCE:-0}"
FORCE_INCOMPLETE="${FORCE_INCOMPLETE:-0}"
RESUME_INCOMPLETE="${RESUME_INCOMPLETE:-0}"

if [ ! -f "$CONFIG_CSV" ]; then
  echo "Missing config CSV: $CONFIG_CSV" >&2
  exit 1
fi

echo "Submitting long building segmentation campaign."
echo "Config: $CONFIG_CSV"
echo "FORCE=${FORCE} FORCE_INCOMPLETE=${FORCE_INCOMPLETE} RESUME_INCOMPLETE=${RESUME_INCOMPLETE}"
echo

tail -n +2 "$CONFIG_CSV" | while IFS=, read -r \
  experiment model train_csv val_csv test_csv input_mode loss augment_mode sampler \
  sampler_alpha lr image_size batch_size epochs time_limit num_workers
do
  [ -n "$experiment" ] || continue
  job_name="blong_${experiment:0:70}"
  export_arg="ALL,CONFIG_CSV=${CONFIG_CSV},FORCE=${FORCE},FORCE_INCOMPLETE=${FORCE_INCOMPLETE},RESUME_INCOMPLETE=${RESUME_INCOMPLETE},EXPERIMENT=${experiment},MODEL=${model},TRAIN_CSV=${train_csv},VAL_CSV=${val_csv},TEST_CSV=${test_csv},INPUT_MODE=${input_mode},LOSS=${loss},AUGMENT_MODE=${augment_mode},SAMPLER=${sampler},SAMPLER_ALPHA=${sampler_alpha},LR=${lr},IMAGE_SIZE=${image_size},BATCH_SIZE=${batch_size},EPOCHS=${epochs},NUM_WORKERS=${num_workers}"
  result="$(
    sbatch \
      --time="$time_limit" \
      --job-name="$job_name" \
      --export="$export_arg" \
      slurm/run_building_long_config.sh
  )"
  echo "${experiment} -> ${result}"
done

echo
echo "Submitted long building rows. Use email notifications and log files instead of frequent scheduler polling."
