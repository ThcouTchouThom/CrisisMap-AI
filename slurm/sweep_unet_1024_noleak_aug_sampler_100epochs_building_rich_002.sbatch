#!/usr/bin/env bash
#SBATCH --job-name=aftermath_aug_br002
#SBATCH --account=def-zonata_gpu
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/home/tgrjlt2/scratch/CrisisMap-AI/logs/%x-%j.out
#SBATCH --error=/home/tgrjlt2/scratch/CrisisMap-AI/logs/%x-%j.err
#SBATCH --mail-user=t.gourjault@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

# Email notifications to avoid frequent scheduler polling

# Campaign 2 augmentation/sampler sweep for splits_noleak_building_rich_002.
# Retry incomplete folders: sbatch --export=ALL,FORCE_INCOMPLETE=1 slurm/sweep_unet_1024_noleak_aug_sampler_100epochs_building_rich_002.sbatch

set -euo pipefail
cd "${HOME}/work/CrisisMap-AI"

export TARGET_SPLIT="splits_noleak_building_rich_002"
export SPLIT_ALIAS="building_rich_002"
export PART_SUMMARY_CSV="outputs/predictions/unet_1024_noleak_aug_sampler_100epochs_building_rich_002_summary.csv"

bash slurm/run_noleak_aug_sampler_split.sh
