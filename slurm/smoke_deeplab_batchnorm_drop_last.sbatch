#!/usr/bin/env bash
#SBATCH --job-name=smoke_dl_bn
#SBATCH --account=def-zonata_gpu
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:10:00
#SBATCH --output=/home/tgrjlt2/scratch/CrisisMap-AI/logs/%x-%j.out
#SBATCH --error=/home/tgrjlt2/scratch/CrisisMap-AI/logs/%x-%j.err
#SBATCH --mail-user=t.gourjault@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

# Email notifications to avoid frequent scheduler polling.
# Smoke test for the DeepLabV3+ BatchNorm failure mode. The real fix is
# drop_last=True on the training DataLoader, so DeepLab never receives the final
# batch_size=1 train batch. This job only checks batch_size=2 train-mode forwards.

set -euo pipefail
cd "${HOME}/work/CrisisMap-AI"
export PYTHONPATH="${PWD}/src:${PWD}:${PYTHONPATH:-}"

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
import torch
import segmentation_models_pytorch as smp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

for model_name, encoder_name in [
    ("deeplabv3plus_resnet50", "resnet50"),
    ("deeplabv3plus_effb3", "efficientnet-b3"),
]:
    print()
    print(f"Testing {model_name} train-mode forward with batch_size=2")
    model = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=1,
    ).to(device)
    model.train()
    x = torch.randn(2, 3, 1024, 1024, device=device)
    target = torch.rand(2, 1, 1024, 1024, device=device)
    output = model(x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(output, target)
    loss.backward()
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Dummy loss: {float(loss.detach().cpu()):.6f}")
    del model, x, target, output, loss
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print()
print("DeepLabV3+ batch_size=2 train-mode smoke test completed.")
PY
