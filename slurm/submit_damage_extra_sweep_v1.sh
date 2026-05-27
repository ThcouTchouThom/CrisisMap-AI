#!/usr/bin/env bash

set -euo pipefail
cd "${HOME}/work/CrisisMap-AI"

CONFIG_CSV="${CONFIG_CSV:-configs/damage_extra_sweep_v1.csv}"
WAIT_FOR_BUILDING100="${WAIT_FOR_BUILDING100:-0}"
BUILDING100_DEPENDENCIES="${BUILDING100_DEPENDENCIES:-}"

if [ ! -f "$CONFIG_CSV" ]; then
  echo "Missing config CSV: $CONFIG_CSV" >&2
  exit 1
fi

dependency_args=()
if [ "$WAIT_FOR_BUILDING100" = "1" ]; then
  if [ -z "$BUILDING100_DEPENDENCIES" ]; then
    echo "WAIT_FOR_BUILDING100=1 requires BUILDING100_DEPENDENCIES=<jobid[:jobid...]>" >&2
    exit 1
  fi
  dependency_args=(--dependency="afterany:${BUILDING100_DEPENDENCIES}")
  echo "Submitting with dependency afterany:${BUILDING100_DEPENDENCIES}"
else
  echo "Submitting independently with no dependencies."
fi

echo "Config: $CONFIG_CSV"
echo

tail -n +2 "$CONFIG_CSV" | while IFS=, read -r \
  experiment split train_csv val_csv test_csv image_size batch_size loss class_weights \
  lr epochs augment_mode sampler damage_sampling_alpha augment_prob \
  damage_augment_threshold high_damage_threshold time_limit num_workers
do
  [ -n "$experiment" ] || continue
  job_name="dmg_extra_${experiment:0:60}"
  export_arg="ALL,CONFIG_CSV=${CONFIG_CSV},EXPERIMENT=${experiment},SPLIT=${split},TRAIN_CSV=${train_csv},VAL_CSV=${val_csv},TEST_CSV=${test_csv},IMAGE_SIZE=${image_size},BATCH_SIZE=${batch_size},LOSS=${loss},CLASS_WEIGHTS=${class_weights},LR=${lr},EPOCHS=${epochs},AUGMENT_MODE=${augment_mode},SAMPLER=${sampler},DAMAGE_SAMPLING_ALPHA=${damage_sampling_alpha},AUGMENT_PROB=${augment_prob},DAMAGE_AUGMENT_THRESHOLD=${damage_augment_threshold},HIGH_DAMAGE_THRESHOLD=${high_damage_threshold},NUM_WORKERS=${num_workers}"
  result="$(
    sbatch \
      "${dependency_args[@]}" \
      --time="$time_limit" \
      --job-name="$job_name" \
      --export="$export_arg" \
      slurm/run_damage_extra_config.sh
  )"
  echo "${experiment} -> ${result}"
done

echo
echo "Submitted damage extra sweep rows. Use email notifications and log files instead of frequent scheduler polling."
echo "Optional dependency usage:"
echo "  WAIT_FOR_BUILDING100=1 BUILDING100_DEPENDENCIES=<jobid[:jobid...]> bash slurm/submit_damage_extra_sweep_v1.sh"
