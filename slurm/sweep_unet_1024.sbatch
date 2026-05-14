#!/usr/bin/env bash
#SBATCH --job-name=aftermath_sweep1024
#SBATCH --account=def-zonata_gpu
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/home/tgrjlt2/scratch/CrisisMap-AI/logs/%x-%j.out
#SBATCH --error=/home/tgrjlt2/scratch/CrisisMap-AI/logs/%x-%j.err

# Sequential 1024 U-Net screening sweep on one H100 GPU.
# To force reruns: sbatch --export=ALL,FORCE=1 slurm/sweep_unet_1024.sbatch

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

FORCE="${FORCE:-0}"
SUMMARY_CSV="outputs/predictions/unet_1024_sweep_summary.csv"

CANDIDATES=(
  "unet_1024_ce_dice_w005_1_4_full_50epochs|ce-dice|0.05 1.0 4.0|1e-4|50"
  "unet_1024_ce_dice_w01_1_4_full_50epochs|ce-dice|0.10 1.0 4.0|1e-4|50"
  "unet_1024_ce_dice_w005_1_3_full_50epochs|ce-dice|0.05 1.0 3.0|1e-4|50"
  "unet_1024_ce_dice_w005_1_5_full_50epochs|ce-dice|0.05 1.0 5.0|1e-4|50"
  "unet_1024_ce_dice_w005_1_4_lr5e5_full_50epochs|ce-dice|0.05 1.0 4.0|5e-5|50"
  "unet_1024_weighted_ce_w005_1_4_full_50epochs|weighted-ce|0.05 1.0 4.0|1e-4|50"
)

train_candidate() {
  local name="$1"
  local loss="$2"
  local weights="$3"
  local lr="$4"
  local epochs="$5"
  local output_dir="outputs/checkpoints/${name}"
  local weights_array

  read -r -a weights_array <<< "$weights"

  python -m src.crisismap.training.train_unet \
    --root data/raw/xbd/train \
    --train-csv data/processed/splits_full/train_pairs.csv \
    --val-csv data/processed/splits_full/val_pairs.csv \
    --output-dir "$output_dir" \
    --image-size 1024 \
    --batch-size 1 \
    --epochs "$epochs" \
    --target-mode 3-class \
    --loss "$loss" \
    --class-weights "${weights_array[@]}" \
    --lr "$lr" \
    --num-workers 8
}

evaluate_candidate() {
  local name="$1"
  local checkpoint="outputs/checkpoints/${name}/best_unet.pt"
  local metrics_json="outputs/predictions/${name}_test_metrics.json"

  if [ ! -f "$checkpoint" ]; then
    echo "Missing checkpoint for evaluation: $checkpoint" >&2
    return 1
  fi

  python -m src.crisismap.evaluation.evaluate_unet \
    --root data/raw/xbd/train \
    --split-csv data/processed/splits_full/test_pairs.csv \
    --checkpoint "$checkpoint" \
    --output "$metrics_json" \
    --image-size 1024 \
    --batch-size 1 \
    --target-mode 3-class \
    --num-workers 8
}

history_complete() {
  local history_json="$1"
  local expected_epochs="$2"

  if [ ! -f "$history_json" ]; then
    return 1
  fi

  python - "$history_json" "$expected_epochs" <<'PY'
import json
import sys
from pathlib import Path

history_path = Path(sys.argv[1])
expected_epochs = int(sys.argv[2])

try:
    with history_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
except Exception:
    sys.exit(1)

if isinstance(data, list):
    history = data
elif isinstance(data, dict) and isinstance(data.get("history"), list):
    history = data["history"]
else:
    sys.exit(1)

if len(history) < expected_epochs:
    sys.exit(1)

epochs = []
for item in history:
    if isinstance(item, dict) and "epoch" in item:
        try:
            epochs.append(int(item["epoch"]))
        except (TypeError, ValueError):
            pass

if epochs and max(epochs) < expected_epochs:
    sys.exit(1)

sys.exit(0)
PY
}

mkdir -p outputs/checkpoints outputs/predictions

for candidate in "${CANDIDATES[@]}"; do
  IFS="|" read -r name loss weights lr epochs <<< "$candidate"
  output_dir="outputs/checkpoints/${name}"
  checkpoint="${output_dir}/best_unet.pt"
  history_json="${output_dir}/metrics_history.json"
  metrics_json="outputs/predictions/${name}_test_metrics.json"
  run_log="${HOME}/scratch/CrisisMap-AI/run_logs/${name}-${SLURM_JOB_ID}.log"

  {
    echo
    echo "==> Candidate: $name"
    echo "loss=$loss weights=$weights lr=$lr epochs=$epochs"
    echo "started=$(date)"

    if [ "$FORCE" != "1" ] && [ -f "$metrics_json" ]; then
      echo "Metrics JSON exists; skipping candidate: $metrics_json"
      exit 0
    fi

    if [ "$FORCE" = "1" ]; then
      echo "FORCE=1: removing previous output and metrics before fresh training."
      rm -rf -- "$output_dir"
      rm -f -- "$metrics_json"
    fi

    if [ "$FORCE" != "1" ] && [ -d "$output_dir" ] && [ ! -f "$metrics_json" ]; then
      if history_complete "$history_json" "$epochs" && [ -f "$checkpoint" ]; then
        echo "Training history shows ${epochs} epochs and best checkpoint exists; running evaluation only."
        evaluate_candidate "$name"
        exit 0
      fi

      echo "WARNING: Candidate appears incomplete: $output_dir"
      echo "WARNING: Not evaluating best_unet.pt because it may come from partial training."
      echo "WARNING: Relaunch with:"
      echo "  sbatch --export=ALL,FORCE=1 slurm/sweep_unet_1024.sbatch"
      exit 0
    fi

    train_candidate "$name" "$loss" "$weights" "$lr" "$epochs"
    evaluate_candidate "$name"
    echo "finished=$(date)"
  } 2>&1 | tee "$run_log"
done

python - <<'PY'
import csv
import json
from pathlib import Path

candidates = [
    ("unet_1024_ce_dice_w005_1_4_full_50epochs", "ce-dice", "0.05 1.0 4.0", "1e-4", 50),
    ("unet_1024_ce_dice_w01_1_4_full_50epochs", "ce-dice", "0.10 1.0 4.0", "1e-4", 50),
    ("unet_1024_ce_dice_w005_1_3_full_50epochs", "ce-dice", "0.05 1.0 3.0", "1e-4", 50),
    ("unet_1024_ce_dice_w005_1_5_full_50epochs", "ce-dice", "0.05 1.0 5.0", "1e-4", 50),
    ("unet_1024_ce_dice_w005_1_4_lr5e5_full_50epochs", "ce-dice", "0.05 1.0 4.0", "5e-5", 50),
    ("unet_1024_weighted_ce_w005_1_4_full_50epochs", "weighted-ce", "0.05 1.0 4.0", "1e-4", 50),
]

def value(values, index):
    if not isinstance(values, list) or len(values) <= index:
        return None
    return values[index]

rows = []
for name, loss, weights, lr, epochs in candidates:
    metrics_path = Path("outputs/predictions") / f"{name}_test_metrics.json"
    checkpoint_path = Path("outputs/checkpoints") / name / "best_unet.pt"
    if not metrics_path.exists():
        print(f"Skipping summary row; missing metrics: {metrics_path}")
        continue
    with metrics_path.open("r", encoding="utf-8") as file:
        metrics = json.load(file)

    iou = metrics.get("iou_per_class", [])
    precision = metrics.get("precision_per_class", [])
    recall = metrics.get("recall_per_class", [])
    f1 = metrics.get("f1_per_class", [])

    rows.append({
        "experiment": name,
        "loss": loss,
        "class_weights": weights,
        "lr": lr,
        "epochs": epochs,
        "image_size": 1024,
        "pixel_accuracy": metrics.get("pixel_accuracy"),
        "mean_iou": metrics.get("mean_iou"),
        "iou_background": value(iou, 0),
        "iou_no_damage": value(iou, 1),
        "iou_damaged": value(iou, 2),
        "precision_damaged": value(precision, 2),
        "recall_damaged": value(recall, 2),
        "f1_damaged": value(f1, 2),
        "checkpoint": str(checkpoint_path),
        "metrics_json": str(metrics_path),
    })

output = Path("outputs/predictions/unet_1024_sweep_summary.csv")
output.parent.mkdir(parents=True, exist_ok=True)
fieldnames = [
    "experiment",
    "loss",
    "class_weights",
    "lr",
    "epochs",
    "image_size",
    "pixel_accuracy",
    "mean_iou",
    "iou_background",
    "iou_no_damage",
    "iou_damaged",
    "precision_damaged",
    "recall_damaged",
    "f1_damaged",
    "checkpoint",
    "metrics_json",
]
with output.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved summary CSV: {output}")
rows.sort(
    key=lambda row: (
        row["iou_damaged"] or -1,
        row["f1_damaged"] or -1,
        row["mean_iou"] or -1,
    ),
    reverse=True,
)
print("\nSorted summary:")
print(f"{'experiment':60s} {'mean_iou':>10s} {'iou_dmg':>10s} {'f1_dmg':>10s}")
for row in rows:
    print(
        f"{row['experiment']:60s} "
        f"{row['mean_iou'] or 0:10.4f} "
        f"{row['iou_damaged'] or 0:10.4f} "
        f"{row['f1_damaged'] or 0:10.4f}"
    )
PY
