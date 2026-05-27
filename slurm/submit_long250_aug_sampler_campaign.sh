#!/usr/bin/env bash

set -euo pipefail
cd "${HOME}/work/CrisisMap-AI"

scripts=(
  "slurm/long250_unet_1024_match_hist1000_aug_none_sampler_none.sbatch"
  "slurm/long250_unet_1024_match_hist1000_aug_damage_aware_sampler_none.sbatch"
  "slurm/long250_unet_1024_match_hist_all_aug_damage_aware_sampler_none.sbatch"
  "slurm/long250_unet_1024_dmg001_v2_aug_damage_aware_sampler_none.sbatch"
  "slurm/long250_unet_1024_match_hist_all_aug_safe_sampler_damage_sqrt_alpha4.sbatch"
)

echo "Submitting long250 augmentation/sampler campaign jobs independently."
echo "No dependencies are used by default."
echo

for script in "${scripts[@]}"; do
  result="$(sbatch "$script")"
  echo "${script} -> ${result}"
done
