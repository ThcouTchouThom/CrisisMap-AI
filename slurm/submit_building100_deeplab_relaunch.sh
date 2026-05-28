#!/usr/bin/env bash

set -euo pipefail
cd "${HOME}/work/CrisisMap-AI"

CONFIG_CSV="${CONFIG_CSV:-configs/building100_deeplab_relaunch.csv}"

if [ ! -f "$CONFIG_CSV" ]; then
  echo "Missing config CSV: $CONFIG_CSV" >&2
  exit 1
fi

echo "Submitting only the 14 failed Building100 DeepLabV3+ rows."
echo "Config: $CONFIG_CSV"
echo "FORCE_INCOMPLETE=1 is exported to clear stale failed DeepLab folders."
echo

tail -n +2 "$CONFIG_CSV" | while IFS=, read -r \
  experiment model train_csv val_csv test_csv input_mode loss augment_mode sampler \
  sampler_alpha lr image_size batch_size epochs time_limit num_workers
do
  [ -n "$experiment" ] || continue
  if [[ "$model" != deeplabv3plus* ]]; then
    echo "Refusing to submit non-DeepLab row: ${experiment} (${model})" >&2
    exit 1
  fi

  job_name="b100_dl_${experiment:0:60}"
  export_arg="ALL,CONFIG_CSV=${CONFIG_CSV},FORCE=0,FORCE_INCOMPLETE=1,RESUME_INCOMPLETE=0,EXPERIMENT=${experiment},MODEL=${model},TRAIN_CSV=${train_csv},VAL_CSV=${val_csv},TEST_CSV=${test_csv},INPUT_MODE=${input_mode},LOSS=${loss},AUGMENT_MODE=${augment_mode},SAMPLER=${sampler},SAMPLER_ALPHA=${sampler_alpha},LR=${lr},IMAGE_SIZE=${image_size},BATCH_SIZE=${batch_size},EPOCHS=${epochs},NUM_WORKERS=${num_workers}"
  result="$(
    sbatch \
      --time="$time_limit" \
      --job-name="$job_name" \
      --export="$export_arg" \
      slurm/run_building100_config.sh
  )"
  echo "${experiment} -> ${result}"
done

echo
echo "Submitted DeepLabV3+ relaunch rows only."
echo "Use email notifications and log files instead of frequent scheduler polling."
