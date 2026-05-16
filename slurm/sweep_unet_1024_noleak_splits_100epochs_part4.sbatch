#!/usr/bin/env bash
#SBATCH --job-name=aftermath_splits100_p4
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

# Part 4 of the no-leak 1024 split sweep.
# Retry incomplete folders: sbatch --export=ALL,FORCE_INCOMPLETE=1 slurm/sweep_unet_1024_noleak_splits_100epochs_part4.sbatch

set -euo pipefail
cd "${HOME}/work/CrisisMap-AI"

export IMAGE_SIZE="${IMAGE_SIZE:-1024}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export LOSS="${LOSS:-ce-dice}"
export CLASS_WEIGHTS="${CLASS_WEIGHTS:-0.05 1.0 4.0}"
export LR="${LR:-1e-4}"
export EPOCHS="${EPOCHS:-100}"
export AUGMENT_MODE="${AUGMENT_MODE:-none}"
export SAMPLER="${SAMPLER:-none}"
export NOLEAK_SPLITS="splits_noleak_match_hist1000 splits_noleak_high_damage_focus"
export PART_SUMMARY_CSV="outputs/predictions/unet_1024_noleak_splits_100epochs_part4_summary.csv"

bash slurm/sweep_unet_1024_noleak_splits_100epochs.sbatch
