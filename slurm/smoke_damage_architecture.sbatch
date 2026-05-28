#!/usr/bin/env bash
#SBATCH --job-name=smoke_dmg_arch
#SBATCH --account=def-zonata_gpu
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=/home/tgrjlt2/scratch/CrisisMap-AI/logs/%x-%j.out
#SBATCH --error=/home/tgrjlt2/scratch/CrisisMap-AI/logs/%x-%j.err
#SBATCH --mail-user=t.gourjault@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

# Email notifications to avoid frequent scheduler polling.
# Smoke test only: instantiate each configured architecture and run one dummy
# forward pass. This does not touch data, checkpoints, or predictions.

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

python - <<'PY'
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path("src").resolve()))
from crisismap.models.damage_model_factory import create_damage_model

config_path = Path("configs/damage_arch_sweep_v1.csv")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Config: {config_path}")

seen = {}
with config_path.open(encoding="utf-8-sig", newline="") as handle:
    for row in csv.DictReader(handle):
        model_name = row["model"].strip()
        if model_name not in seen:
            seen[model_name] = int(row.get("base_channels") or 32)

for model_name, base_channels in seen.items():
    print()
    print(f"Testing {model_name} (base_channels={base_channels})")
    model = create_damage_model(
        model_name,
        num_classes=3,
        in_channels=6,
        base_channels=base_channels,
    ).to(device)
    model.eval()
    x = torch.randn(1, 6, 256, 256, device=device)
    with torch.no_grad():
        y = model(x)
    print(f"Output shape: {tuple(y.shape)}")
    if tuple(y.shape) != (1, 3, 256, 256):
        raise SystemExit(f"Unexpected output shape for {model_name}: {tuple(y.shape)}")
    del model, x, y
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print()
print("All configured damage architectures passed the dummy forward smoke test.")
PY
