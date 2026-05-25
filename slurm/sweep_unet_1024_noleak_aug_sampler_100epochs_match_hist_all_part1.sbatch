#!/usr/bin/env bash
#SBATCH --job-name=aftermath_aug_histall_p1
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

# Campaign 2 part 1 for splits_noleak_match_hist_all.
# The old 8-variant wrappers can exceed the time limit; use part jobs for safer relaunches.
# Retry incomplete folders: sbatch --export=ALL,FORCE_INCOMPLETE=1 slurm/sweep_unet_1024_noleak_aug_sampler_100epochs_match_hist_all_part1.sbatch

set -euo pipefail
cd "${HOME}/work/CrisisMap-AI"

export TARGET_SPLIT="splits_noleak_match_hist_all"
export SPLIT_ALIAS="match_hist_all"
export AUG_SAMPLER_COMBOS="none|none|4 safe|none|4 damage-aware|none|4 none|damage-simple|4"
export PART_SUMMARY_CSV="outputs/predictions/unet_1024_noleak_aug_sampler_100epochs_match_hist_all_part1_summary.csv"

bash slurm/run_noleak_aug_sampler_split.sh
