#!/usr/bin/env bash
#SBATCH --job-name=aftermath_u1024
#SBATCH --account=def-zonata_gpu
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/home/tgrjlt2/scratch/CrisisMap-AI/logs/%x-%j.out
#SBATCH --error=/home/tgrjlt2/scratch/CrisisMap-AI/logs/%x-%j.err

# First serious Alliance/Rorqual U-Net baseline:
# 1024x1024, full-data splits, 100 epochs, weighted CE + Dice.

set -euo pipefail
cd "${HOME}/work/CrisisMap-AI"

module --force purge
module load StdEnv/2023
module load python/3.11
module load gcc
module load arrow/23.0.1
module load cuda
module load opencv/4.13.0

source "${HOME}/virtualenvs/crisismap-ai/bin/activate"
export TRITON_CACHE_DIR="${HOME}/scratch/CrisisMap-AI/triton_cache"
mkdir -p "${TRITON_CACHE_DIR}" "${HOME}/scratch/CrisisMap-AI/run_logs"

EXPERIMENT="unet_1024_ce_dice_w005_1_4_full_100epochs"
RUN_LOG="${HOME}/scratch/CrisisMap-AI/run_logs/${EXPERIMENT}-${SLURM_JOB_ID}.log"

{
  echo "Experiment: ${EXPERIMENT}"
  echo "Started: $(date)"
  echo "Host: $(hostname)"

  python -m src.crisismap.training.train_unet \
    --root data/raw/xbd/train \
    --train-csv data/processed/splits_full/train_pairs.csv \
    --val-csv data/processed/splits_full/val_pairs.csv \
    --output-dir "outputs/checkpoints/${EXPERIMENT}" \
    --image-size 1024 \
    --batch-size 1 \
    --epochs 100 \
    --target-mode 3-class \
    --loss ce-dice \
    --class-weights 0.05 1.0 4.0 \
    --lr 1e-4 \
    --num-workers 8

  python -m src.crisismap.evaluation.evaluate_unet \
    --root data/raw/xbd/train \
    --split-csv data/processed/splits_full/test_pairs.csv \
    --checkpoint "outputs/checkpoints/${EXPERIMENT}/best_unet.pt" \
    --output "outputs/predictions/${EXPERIMENT}_test_metrics.json" \
    --image-size 1024 \
    --batch-size 1 \
    --target-mode 3-class \
    --num-workers 8

  echo "Finished: $(date)"
} 2>&1 | tee "$RUN_LOG"
