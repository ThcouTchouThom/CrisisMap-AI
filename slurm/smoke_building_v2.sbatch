#!/usr/bin/env bash
#SBATCH --job-name=smoke_building_v2
#SBATCH --account=def-zonata_gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/tgrjlt2/CrisisMap-AI/logs/%x-%j.out
#SBATCH --error=/scratch/tgrjlt2/CrisisMap-AI/logs/%x-%j.err
# Email notifications to avoid frequent scheduler polling
#SBATCH --mail-user=t.gourjault@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

set -euo pipefail

: "${SCRATCH:?SCRATCH environment variable is required}"

CODE_DIR="${CODE_DIR:-${HOME}/work/CrisisMap-AI}"
VENV_PATH="${VENV_PATH:-${HOME}/virtualenvs/crisismap-ai/bin/activate}"

export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning}"
export TRITON_CACHE_DIR="${SCRATCH}/CrisisMap-AI/triton_cache"
mkdir -p "${TRITON_CACHE_DIR}" "${SCRATCH}/CrisisMap-AI/logs" "${SCRATCH}/CrisisMap-AI/run_logs"

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
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path("scripts").resolve()))
from train_building_segmentation import build_loss, build_model, input_channels

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print("gpu:", torch.cuda.get_device_name(0))

models = [
    "unetplusplus_effb4",
    "unetplusplus_effb3",
    "deeplabv3plus_resnet50",
]
losses = [
    "focal-tversky",
    "focal-tversky-boundary",
    "focal-dice",
    "bce-dice",
]
x = torch.randn(2, 3, 512, 512, device=device)
y = torch.randint(0, 2, (2, 1, 512, 512), device=device).float()
for model_name in models:
    model, actual = build_model(model_name, input_channels("pre"), device)
    model.train()
    out = model(x)
    print(model_name, actual, tuple(out.shape))
    assert tuple(out.shape) == (2, 1, 512, 512)
    for loss_name in losses:
        args = argparse.Namespace(
            loss=loss_name,
            focal_alpha=0.25,
            focal_gamma=2.0,
            focal_tversky_alpha=0.3,
            focal_tversky_beta=0.7,
            focal_tversky_gamma=0.75,
            boundary_loss_weight=0.2,
        )
        loss = build_loss(args)(out, y)
        print("  loss", loss_name, float(loss.detach().cpu()))
        assert torch.isfinite(loss)
    del model, out
    if device.type == "cuda":
        torch.cuda.empty_cache()

print("Building v2 smoke test OK")
PY
