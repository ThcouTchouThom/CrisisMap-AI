#!/usr/bin/env bash
#SBATCH --job-name=aft_l250_histall_sqrt
#SBATCH --account=def-zonata_gpu
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=05:30:00
#SBATCH --output=/home/tgrjlt2/scratch/CrisisMap-AI/logs/%x-%j.out
#SBATCH --error=/home/tgrjlt2/scratch/CrisisMap-AI/logs/%x-%j.err
#SBATCH --mail-user=t.gourjault@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

# Email notifications to avoid frequent scheduler polling.
# 250-epoch long run estimated from the completed 100-epoch campaign.

set -euo pipefail
cd "${HOME}/work/CrisisMap-AI"

export TARGET_SPLIT="splits_noleak_match_hist_all"
export SPLIT_ALIAS="match_hist_all"
export AUGMENT_MODE="safe"
export SAMPLER="damage-sqrt"
export DAMAGE_SAMPLING_ALPHA="4"

bash slurm/run_long250_aug_sampler_job.sh
