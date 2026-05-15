#!/usr/bin/env bash
#SBATCH --job-name=aftermath_splits100
#SBATCH --account=def-zonata_gpu
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/home/tgrjlt2/scratch/CrisisMap-AI/logs/%x-%j.out
#SBATCH --error=/home/tgrjlt2/scratch/CrisisMap-AI/logs/%x-%j.err

# No-leak split sweep for 1024 U-Net.
# Force reruns: sbatch --export=ALL,FORCE=1 slurm/sweep_unet_1024_noleak_splits_100epochs.sbatch

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
mkdir -p outputs/checkpoints outputs/predictions

FORCE="${FORCE:-0}"
JOB_ID="${SLURM_JOB_ID:-manual}"

IMAGE_SIZE="${IMAGE_SIZE:-1024}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LOSS="${LOSS:-ce-dice}"
CLASS_WEIGHTS="${CLASS_WEIGHTS:-0.05 1.0 4.0}"
LR="${LR:-1e-4}"
EPOCHS="${EPOCHS:-100}"
AUGMENT_MODE="${AUGMENT_MODE:-none}"
AUGMENT_PROB="${AUGMENT_PROB:-0.5}"
DAMAGE_AUGMENT_THRESHOLD="${DAMAGE_AUGMENT_THRESHOLD:-0.001}"
SAMPLER="${SAMPLER:-none}"
DAMAGE_SAMPLING_ALPHA="${DAMAGE_SAMPLING_ALPHA:-4.0}"
HIGH_DAMAGE_THRESHOLD="${HIGH_DAMAGE_THRESHOLD:-0.02}"
SUMMARY_CSV="outputs/predictions/unet_1024_noleak_splits_100epochs_summary.csv"

SPLITS=(
  "splits_noleak_full_train"
  "splits_noleak_full_balanced_damage"
  "splits_noleak_match_hist1500"
  "splits_noleak_match_hist_all"
  "splits_noleak_dmg001_v2"
  "splits_noleak_damage_heavy"
  "splits_noleak_disaster_stratified_150"
  "splits_noleak_disaster_stratified_200"
  "splits_noleak_building_rich_002"
  "splits_noleak_building_rich_003"
)

split_has_files() {
  local split_dir="$1"
  [ -f "data/processed/${split_dir}/train_pairs.csv" ] &&
    [ -f "data/processed/${split_dir}/val_pairs.csv" ] &&
    [ -f "data/processed/${split_dir}/test_pairs.csv" ]
}

ensure_splits() {
  if [ ! -f "data/processed/xbd_train_index.csv" ] || ! split_has_files "splits_full"; then
    echo "Missing index or common splits. Run: bash slurm/setup_rorqual.sh" >&2
    return 1
  fi

  for split_dir in "${SPLITS[@]}"; do
    if ! split_has_files "$split_dir"; then
      echo "Advanced no-leak splits missing; creating them."
      python scripts/create_advanced_noleak_train_splits.py \
        --index data/processed/xbd_train_index.csv \
        --common-split-dir data/processed/splits_full \
        --output-root data/processed \
        --seed 42
      break
    fi
  done

  for split_dir in "${SPLITS[@]}"; do
    if ! split_has_files "$split_dir"; then
      echo "Missing split files after creation: data/processed/${split_dir}" >&2
      return 1
    fi
  done
}

history_complete() {
  local history_json="$1"
  local expected_epochs="$2"
  [ -f "$history_json" ] || return 1
  python - "$history_json" "$expected_epochs" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = int(sys.argv[2])
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    sys.exit(1)
history = data if isinstance(data, list) else data.get("history") if isinstance(data, dict) else None
if not isinstance(history, list) or len(history) < expected:
    sys.exit(1)
epochs = []
for item in history:
    if isinstance(item, dict) and "epoch" in item:
        try:
            epochs.append(int(item["epoch"]))
        except (TypeError, ValueError):
            pass
if epochs and max(epochs) < expected:
    sys.exit(1)
sys.exit(0)
PY
}

experiment_name() {
  local split_dir="$1"
  echo "unet_${IMAGE_SIZE}_${split_dir}_${EPOCHS}epochs_aug-${AUGMENT_MODE}_sampler-${SAMPLER}"
}

train_and_eval_split() {
  local split_dir="$1"
  local name
  local output_dir
  local history_json
  local checkpoint
  local metrics_json
  local weights_array

  name="$(experiment_name "$split_dir")"
  output_dir="outputs/checkpoints/${name}"
  history_json="${output_dir}/metrics_history.json"
  checkpoint="${output_dir}/best_unet.pt"
  metrics_json="outputs/predictions/${name}_test_metrics.json"
  read -r -a weights_array <<< "$CLASS_WEIGHTS"

  echo
  echo "==> Split: ${split_dir}"
  echo "experiment=${name}"

  if [ "$FORCE" = "1" ]; then
    rm -rf -- "$output_dir"
    rm -f -- "$metrics_json"
  elif history_complete "$history_json" "$EPOCHS" && [ -f "$metrics_json" ]; then
    echo "Complete run found; skipping."
    return 0
  elif [ -d "$output_dir" ]; then
    if history_complete "$history_json" "$EPOCHS" && [ -f "$checkpoint" ]; then
      echo "Training complete; evaluating missing metrics."
    else
      echo "WARNING: Incomplete output folder, not evaluating partial checkpoint: $output_dir"
      echo "WARNING: Relaunch with FORCE=1 to retrain."
      return 0
    fi
  fi

  if ! history_complete "$history_json" "$EPOCHS"; then
    python -m src.crisismap.training.train_unet \
      --root data/raw/xbd/train \
      --train-csv "data/processed/${split_dir}/train_pairs.csv" \
      --val-csv "data/processed/${split_dir}/val_pairs.csv" \
      --output-dir "$output_dir" \
      --image-size "$IMAGE_SIZE" \
      --batch-size "$BATCH_SIZE" \
      --epochs "$EPOCHS" \
      --target-mode 3-class \
      --loss "$LOSS" \
      --class-weights "${weights_array[@]}" \
      --lr "$LR" \
      --num-workers 8 \
      --augment-mode "$AUGMENT_MODE" \
      --augment-prob "$AUGMENT_PROB" \
      --damage-augment-threshold "$DAMAGE_AUGMENT_THRESHOLD" \
      --sampler "$SAMPLER" \
      --damage-sampling-alpha "$DAMAGE_SAMPLING_ALPHA" \
      --high-damage-threshold "$HIGH_DAMAGE_THRESHOLD"
  fi

  if [ ! -f "$metrics_json" ]; then
    python -m src.crisismap.evaluation.evaluate_unet \
      --root data/raw/xbd/train \
      --split-csv "data/processed/${split_dir}/test_pairs.csv" \
      --checkpoint "$checkpoint" \
      --output "$metrics_json" \
      --image-size "$IMAGE_SIZE" \
      --batch-size 1 \
      --target-mode 3-class \
      --num-workers 8
  fi
}

ensure_splits

echo "No-leak split sweep"
echo "IMAGE_SIZE=${IMAGE_SIZE} BATCH_SIZE=${BATCH_SIZE} LOSS=${LOSS} LR=${LR} EPOCHS=${EPOCHS}"
echo "AUGMENT_MODE=${AUGMENT_MODE} SAMPLER=${SAMPLER}"

for split_dir in "${SPLITS[@]}"; do
  name="$(experiment_name "$split_dir")"
  run_log="${HOME}/scratch/CrisisMap-AI/run_logs/${name}-${JOB_ID}.log"
  train_and_eval_split "$split_dir" 2>&1 | tee "$run_log"
done

IMAGE_SIZE="$IMAGE_SIZE" \
BATCH_SIZE="$BATCH_SIZE" \
LOSS="$LOSS" \
CLASS_WEIGHTS="$CLASS_WEIGHTS" \
LR="$LR" \
EPOCHS="$EPOCHS" \
AUGMENT_MODE="$AUGMENT_MODE" \
SAMPLER="$SAMPLER" \
python - <<'PY'
import csv
import json
import os
from pathlib import Path

splits = [
    "splits_noleak_full_train",
    "splits_noleak_full_balanced_damage",
    "splits_noleak_match_hist1500",
    "splits_noleak_match_hist_all",
    "splits_noleak_dmg001_v2",
    "splits_noleak_damage_heavy",
    "splits_noleak_disaster_stratified_150",
    "splits_noleak_disaster_stratified_200",
    "splits_noleak_building_rich_002",
    "splits_noleak_building_rich_003",
]
image_size = os.environ["IMAGE_SIZE"]
batch_size = os.environ["BATCH_SIZE"]
loss = os.environ["LOSS"]
class_weights = os.environ["CLASS_WEIGHTS"]
lr = os.environ["LR"]
epochs = os.environ["EPOCHS"]
augment_mode = os.environ["AUGMENT_MODE"]
sampler = os.environ["SAMPLER"]


def class_value(metrics, key, index):
    values = metrics.get(key, [])
    if not isinstance(values, list) or len(values) <= index:
        return None
    return values[index]


rows = []
for split in splits:
    experiment = f"unet_{image_size}_{split}_{epochs}epochs_aug-{augment_mode}_sampler-{sampler}"
    metrics_path = Path("outputs/predictions") / f"{experiment}_test_metrics.json"
    checkpoint_path = Path("outputs/checkpoints") / experiment / "best_unet.pt"
    if not metrics_path.exists():
        print(f"Skipping summary row; missing metrics: {metrics_path}")
        continue
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows.append({
        "experiment": experiment,
        "split": split,
        "image_size": image_size,
        "batch_size": batch_size,
        "loss": loss,
        "class_weights": class_weights,
        "lr": lr,
        "epochs": epochs,
        "augment_mode": augment_mode,
        "sampler": sampler,
        "test_pixel_accuracy": metrics.get("pixel_accuracy"),
        "test_mean_iou": metrics.get("mean_iou"),
        "test_iou_background": class_value(metrics, "iou_per_class", 0),
        "test_iou_no_damage": class_value(metrics, "iou_per_class", 1),
        "test_iou_damaged": class_value(metrics, "iou_per_class", 2),
        "test_precision_damaged": class_value(metrics, "precision_per_class", 2),
        "test_recall_damaged": class_value(metrics, "recall_per_class", 2),
        "test_f1_damaged": class_value(metrics, "f1_per_class", 2),
        "checkpoint": str(checkpoint_path),
        "test_metrics_json": str(metrics_path),
    })
rows.sort(
    key=lambda row: (
        row["test_iou_damaged"] if row["test_iou_damaged"] is not None else -1,
        row["test_f1_damaged"] if row["test_f1_damaged"] is not None else -1,
        row["test_mean_iou"] if row["test_mean_iou"] is not None else -1,
    ),
    reverse=True,
)
output = Path("outputs/predictions/unet_1024_noleak_splits_100epochs_summary.csv")
output.parent.mkdir(parents=True, exist_ok=True)
fieldnames = [
    "experiment",
    "split",
    "image_size",
    "batch_size",
    "loss",
    "class_weights",
    "lr",
    "epochs",
    "augment_mode",
    "sampler",
    "test_pixel_accuracy",
    "test_mean_iou",
    "test_iou_background",
    "test_iou_no_damage",
    "test_iou_damaged",
    "test_precision_damaged",
    "test_recall_damaged",
    "test_f1_damaged",
    "checkpoint",
    "test_metrics_json",
]
with output.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f"Saved summary CSV: {output}")
for row in rows:
    print(row["experiment"], row["test_iou_damaged"], row["test_f1_damaged"], row["test_mean_iou"])
PY
