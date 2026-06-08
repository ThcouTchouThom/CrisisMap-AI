#!/usr/bin/env bash
#SBATCH --job-name=smoke_dftv2
#SBATCH --account=def-zonata_gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/tgrjlt2/CrisisMap-AI/logs/%x-%j.out
#SBATCH --error=/scratch/tgrjlt2/CrisisMap-AI/logs/%x-%j.err
#SBATCH --mail-user=t.gourjault@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

set -euo pipefail

CODE_DIR="${CODE_DIR:-${HOME}/work/CrisisMap-AI}"
VENV_PATH="${VENV_PATH:-${HOME}/virtualenvs/crisismap-ai/bin/activate}"

if [ -z "${SCRATCH:-}" ]; then
  echo "ERROR: SCRATCH is not set." >&2
  exit 2
fi

export TRITON_CACHE_DIR="${SCRATCH}/CrisisMap-AI/triton_cache"
mkdir -p \
  "${TRITON_CACHE_DIR}" \
  "${SCRATCH}/CrisisMap-AI/run_logs" \
  "${SCRATCH}/CrisisMap-AI/logs"

cd "${CODE_DIR}"
export PYTHONPATH="${CODE_DIR}/src:${CODE_DIR}:${PYTHONPATH:-}"

module --force purge
module load StdEnv/2023
module load python/3.11
module load gcc
module load arrow/23.0.1
module load cuda
module load opencv/4.13.0

source "${VENV_PATH}"

python - <<'PY'
import csv
from pathlib import Path

import torch

from crisismap.models.damage_model_factory import create_damage_model

config = Path("configs/damage_focal_tversky_v2.csv")
with config.open(encoding="utf-8-sig", newline="") as f:
    models = sorted({row["model"] for row in csv.DictReader(f)})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

for name in models:
    print("Testing", name)
    model = create_damage_model(name, num_classes=3, in_channels=6, base_channels=32).to(device)
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 6, 256, 256, device=device)
        y = model(x)
    print("  output:", tuple(y.shape))
    if tuple(y.shape) != (1, 3, 256, 256):
        raise SystemExit(f"Unexpected output shape for {name}: {tuple(y.shape)}")
print("damage_focal_tversky_v2 smoke test OK")
PY
