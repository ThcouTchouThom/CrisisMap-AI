#!/usr/bin/env bash

set -euo pipefail
cd "${HOME}/work/CrisisMap-AI"

CONFIG_CSV="${CONFIG_CSV:-configs/building100_sweep_v1.csv}"
WAIT_FOR_LONG250="${WAIT_FOR_LONG250:-1}"
LONG250_DEPENDENCIES="${LONG250_DEPENDENCIES:-13271923:13271924:13271925:13271926:13271927}"

if [ ! -f "$CONFIG_CSV" ]; then
  echo "Missing config CSV: $CONFIG_CSV" >&2
  exit 1
fi

dependency_args=()
if [ "$WAIT_FOR_LONG250" = "1" ]; then
  dependency_args=(--dependency="afterany:${LONG250_DEPENDENCIES}")
  echo "Submitting with dependency afterany:${LONG250_DEPENDENCIES}"
else
  echo "Submitting without long250 dependency."
fi

echo "Config: $CONFIG_CSV"
echo

tail -n +2 "$CONFIG_CSV" | while IFS=, read -r \
  experiment model train_csv val_csv test_csv input_mode loss augment_mode sampler \
  sampler_alpha lr image_size batch_size epochs time_limit num_workers
do
  [ -n "$experiment" ] || continue
  job_name="b100_${experiment:0:70}"
  export_arg="ALL,CONFIG_CSV=${CONFIG_CSV},EXPERIMENT=${experiment},MODEL=${model},TRAIN_CSV=${train_csv},VAL_CSV=${val_csv},TEST_CSV=${test_csv},INPUT_MODE=${input_mode},LOSS=${loss},AUGMENT_MODE=${augment_mode},SAMPLER=${sampler},SAMPLER_ALPHA=${sampler_alpha},LR=${lr},IMAGE_SIZE=${image_size},BATCH_SIZE=${batch_size},EPOCHS=${epochs},NUM_WORKERS=${num_workers}"
  result="$(
    sbatch \
      "${dependency_args[@]}" \
      --time="$time_limit" \
      --job-name="$job_name" \
      --export="$export_arg" \
      slurm/run_building100_config.sh
  )"
  echo "${experiment} -> ${result}"
done

echo
echo "Submitted Building100 sweep rows. Use email notifications and log files instead of frequent scheduler polling."
