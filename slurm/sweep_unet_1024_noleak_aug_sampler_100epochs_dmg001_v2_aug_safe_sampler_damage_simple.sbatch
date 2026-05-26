#!/usr/bin/env bash
#SBATCH --job-name=aft_aug_d001_safe_simple
#SBATCH --account=def-zonata_gpu
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/home/tgrjlt2/scratch/CrisisMap-AI/logs/%x-%j.out
#SBATCH --error=/home/tgrjlt2/scratch/CrisisMap-AI/logs/%x-%j.err
#SBATCH --mail-user=t.gourjault@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

# Email notifications to avoid frequent scheduler polling
# Single-variant jobs are recommended: dmg001_v2_part1 timed out after one full
# 100-epoch training, before evaluation/summary could finish.

set -euo pipefail
cd "${HOME}/work/CrisisMap-AI"

export TARGET_SPLIT="splits_noleak_dmg001_v2"
export SPLIT_ALIAS="dmg001_v2"
export AUG_SAMPLER_COMBOS="safe|damage-simple|4"
export PART_SUMMARY_CSV="outputs/predictions/unet_1024_noleak_aug_sampler_100epochs_dmg001_v2_aug_safe_sampler_damage_simple_summary.csv"

bash slurm/run_noleak_aug_sampler_split.sh
