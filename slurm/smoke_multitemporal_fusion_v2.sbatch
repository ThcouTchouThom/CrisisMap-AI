#!/bin/bash
#SBATCH --job-name=smoke_mtf_v2
#SBATCH --account=def-zonata_gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=${SCRATCH}/CrisisMap-AI/logs/%x-%j.out
#SBATCH --error=${SCRATCH}/CrisisMap-AI/logs/%x-%j.err
# Email notifications to avoid frequent scheduler polling
#SBATCH --mail-user=t.gourjault@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

set -euo pipefail

if [[ -z "${SCRATCH:-}" ]]; then
  echo "ERROR: SCRATCH is not set." >&2
  exit 2
fi

CODE_DIR="${CODE_DIR:-${HOME}/work/CrisisMap-AI}"
VENV_PATH="${VENV_PATH:-${HOME}/virtualenvs/crisismap-ai/bin/activate}"

export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning}"
export TRITON_CACHE_DIR="${SCRATCH}/CrisisMap-AI/triton_cache"
mkdir -p "${TRITON_CACHE_DIR}" "${SCRATCH}/CrisisMap-AI/logs" "${SCRATCH}/CrisisMap-AI/run_logs"

cd "${CODE_DIR}"

module load StdEnv/2023
module load python/3.11
module load gcc
module load arrow/23.0.1
module load cuda
module load opencv/4.13.0

source "${VENV_PATH}"

python - <<'PY'
import torch

from crisismap.models.multitemporal_fusion import (
    create_multitemporal_fusion_model,
    supported_multitemporal_fusion_models,
)

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print("gpu:", torch.cuda.get_device_name(0))

x = torch.randn(1, 6, 256, 256, device=device)
for name in supported_multitemporal_fusion_models():
    damage_channel_options = (3,)
    if name == "mtf_resnet50_fpn_attention":
        damage_channel_options = (3, 1)
    for damage_channels in damage_channel_options:
        print(f"Testing {name} damage_channels={damage_channels}")
        model = create_multitemporal_fusion_model(
            name,
            damage_channels=damage_channels,
        ).to(device)
        model.eval()
        with torch.no_grad():
            out = model(x)
        building_shape = tuple(out["building_logits"].shape)
        damage_shape = tuple(out["damage_logits"].shape)
        print("  building", building_shape, "damage", damage_shape)
        assert building_shape == (1, 1, 256, 256)
        assert damage_shape == (1, damage_channels, 256, 256)
        del model, out
        if device.type == "cuda":
            torch.cuda.empty_cache()

print("Multi-Temporal Fusion v2 smoke test OK")
PY
