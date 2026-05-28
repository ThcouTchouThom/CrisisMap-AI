#!/usr/bin/env bash

set -euo pipefail
cd "${HOME}/work/CrisisMap-AI"

CONFIG_CSV="${CONFIG_CSV:-configs/damage_arch_sweep_v1.csv}"

if [ ! -f "$CONFIG_CSV" ]; then
  echo "Missing config CSV: $CONFIG_CSV" >&2
  exit 1
fi

echo "Submitting Axis 2 damage architecture sweep independently."
echo "Config: $CONFIG_CSV"
echo

tail -n +2 "$CONFIG_CSV" | while IFS=, read -r \
  experiment model split train_csv val_csv test_csv image_size batch_size epochs loss \
  class_weights lr augment_mode augment_prob damage_augment_threshold sampler \
  damage_sampling_alpha high_damage_threshold base_channels time_limit num_workers status
do
  [ -n "$experiment" ] || continue
  job_name="dmg_arch_${experiment:0:60}"
  export_arg="ALL,EXPERIMENT=${experiment},MODEL=${model},SPLIT=${split},TRAIN_CSV=${train_csv},VAL_CSV=${val_csv},TEST_CSV=${test_csv},IMAGE_SIZE=${image_size},BATCH_SIZE=${batch_size},EPOCHS=${epochs},LOSS=${loss},CLASS_WEIGHTS=${class_weights},LR=${lr},AUGMENT_MODE=${augment_mode},AUGMENT_PROB=${augment_prob},DAMAGE_AUGMENT_THRESHOLD=${damage_augment_threshold},SAMPLER=${sampler},DAMAGE_SAMPLING_ALPHA=${damage_sampling_alpha},HIGH_DAMAGE_THRESHOLD=${high_damage_threshold},BASE_CHANNELS=${base_channels},NUM_WORKERS=${num_workers}"
  result="$(
    sbatch \
      --time="$time_limit" \
      --job-name="$job_name" \
      --export="$export_arg" \
      slurm/run_damage_arch_config.sh
  )"
  echo "${experiment} -> ${result}"
done

echo
echo "Submitted Axis 2 damage architecture rows."
echo "Use email notifications and log files instead of frequent scheduler polling."
