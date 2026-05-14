#!/usr/bin/env bash
#SBATCH --job-name=aftermath_smoke512
#SBATCH --account=def-zonata_gpu
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/tgrjlt2/scratch/CrisisMap-AI/logs/%x-%j.out
#SBATCH --error=/home/tgrjlt2/scratch/CrisisMap-AI/logs/%x-%j.err

# Technical smoke test only. Adjust account, partition, qos, time, or memory
# if your allocation requires different settings.

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

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

python -m src.crisismap.training.train_unet \
  --root data/raw/xbd/train \
  --train-csv data/processed/splits_full/train_pairs.csv \
  --val-csv data/processed/splits_full/val_pairs.csv \
  --output-dir outputs/checkpoints/smoke_unet_512 \
  --image-size 512 \
  --batch-size 2 \
  --epochs 1 \
  --target-mode 3-class \
  --loss ce-dice \
  --class-weights 0.05 1.0 4.0 \
  --lr 1e-4 \
  --max-train-samples 32 \
  --max-val-samples 16 \
  --num-workers 4 \
  2>&1 | tee "${HOME}/scratch/CrisisMap-AI/run_logs/smoke_unet_512-${SLURM_JOB_ID}.log"

test -f outputs/checkpoints/smoke_unet_512/best_unet.pt
ls -lh outputs/checkpoints/smoke_unet_512/best_unet.pt
echo "Smoke test completed: checkpoint created."
