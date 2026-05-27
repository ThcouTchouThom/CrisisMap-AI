#!/usr/bin/env bash
#SBATCH --job-name=smoke_building_env
#SBATCH --account=def-zonata_gpu
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/home/tgrjlt2/scratch/CrisisMap-AI/logs/%x-%j.out
#SBATCH --error=/home/tgrjlt2/scratch/CrisisMap-AI/logs/%x-%j.err
#SBATCH --mail-user=t.gourjault@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

# Email notifications to avoid frequent scheduler polling.
# Smoke test for the Rorqual building segmentation environment. This checks
# CUDA and SMP inside a real GPU allocation, unlike login-node checks.

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
mkdir -p "${TRITON_CACHE_DIR}"

python scripts/check_rorqual_building_env.py

python - <<'PY'
import torch
import timm
import segmentation_models_pytorch as smp

print("Smoke forward test")
print("=" * 30)
print(f"torch: {torch.__version__}")
print(f"timm: {timm.__version__}")
print(f"segmentation_models_pytorch: {smp.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

model = smp.UnetPlusPlus(
    encoder_name="efficientnet-b3",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None,
).to(device)
model.eval()

with torch.no_grad():
    dummy = torch.randn(1, 3, 256, 256, device=device)
    output = model(dummy)

print(f"Dummy input shape: {tuple(dummy.shape)}")
print(f"Output shape: {tuple(output.shape)}")
PY
