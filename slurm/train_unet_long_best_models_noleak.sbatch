#!/usr/bin/env bash
#SBATCH --job-name=aftermath_noleak
#SBATCH --account=def-zonata_gpu
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/home/tgrjlt2/scratch/CrisisMap-AI/logs/%x-%j.out
#SBATCH --error=/home/tgrjlt2/scratch/CrisisMap-AI/logs/%x-%j.err
#SBATCH --mail-user=t.gourjault@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

# Email notifications to avoid frequent scheduler polling

# Leakage-free long training for the best CrisisMap AI U-Net configurations.
# To force reruns: sbatch --export=ALL,FORCE=1 slurm/train_unet_long_best_models_noleak.sbatch

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
SUMMARY_CSV="outputs/predictions/unet_long_best_models_noleak_summary.csv"

CANDIDATES=(
  "unet_1024_ce_dice_w005_1_4_noleak_match_hist1000_bs2_250epochs|1024|splits_noleak_match_hist1000|noleak_match_hist1000|2|0.05 1.0 4.0|250"
  "unet_1024_ce_dice_w004_1_5_noleak_match_hist1000_bs3_250epochs|1024|splits_noleak_match_hist1000|noleak_match_hist1000|3|0.04 1.0 5.0|250"
  "unet_1024_ce_dice_w005_1_4_noleak_match_dmg001_bs3_250epochs|1024|splits_noleak_match_dmg001|noleak_match_dmg001|3|0.05 1.0 4.0|250"
  "unet_512_ce_dice_w005_1_4_noleak_old4_bs2_250epochs|512|splits_noleak_old4|noleak_old4|2|0.05 1.0 4.0|250"
)

split_has_files() {
  local split_dir="$1"
  [ -f "data/processed/${split_dir}/train_pairs.csv" ] &&
    [ -f "data/processed/${split_dir}/val_pairs.csv" ] &&
    [ -f "data/processed/${split_dir}/test_pairs.csv" ]
}

ensure_required_files() {
  if [ ! -f "data/processed/xbd_train_index.csv" ]; then
    echo "Missing index CSV: data/processed/xbd_train_index.csv" >&2
    echo "Run: bash slurm/setup_rorqual.sh" >&2
    return 1
  fi

  if ! split_has_files "splits_full"; then
    echo "Missing common full-data split files under data/processed/splits_full/" >&2
    echo "Run: bash slurm/setup_rorqual.sh" >&2
    return 1
  fi
}

create_noleak_splits_if_needed() {
  if split_has_files "splits_noleak_old4" &&
     split_has_files "splits_noleak_match_hist1000" &&
     split_has_files "splits_noleak_match_dmg001"; then
    return 0
  fi

  echo "No-leak split folders are missing; creating them from xbd_train_index.csv."
  python scripts/create_noleak_common_eval_splits.py \
    --index data/processed/xbd_train_index.csv \
    --common-split-dir data/processed/splits_full \
    --output-root data/processed \
    --seed 42
}

ensure_candidate_splits() {
  local candidate
  local name
  local resolution
  local split_dir
  local split_label
  local batch_size
  local weights
  local epochs

  for candidate in "${CANDIDATES[@]}"; do
    IFS="|" read -r name resolution split_dir split_label batch_size weights epochs <<< "$candidate"
    if ! split_has_files "$split_dir"; then
      echo "Missing split files for candidate '${name}': data/processed/${split_dir}" >&2
      return 1
    fi
  done
}

train_candidate() {
  local name="$1"
  local resolution="$2"
  local split_dir="$3"
  local batch_size="$4"
  local weights="$5"
  local epochs="$6"
  local output_dir="outputs/checkpoints/${name}"
  local weights_array

  read -r -a weights_array <<< "$weights"

  python -m src.crisismap.training.train_unet \
    --root data/raw/xbd/train \
    --train-csv "data/processed/${split_dir}/train_pairs.csv" \
    --val-csv "data/processed/${split_dir}/val_pairs.csv" \
    --output-dir "$output_dir" \
    --image-size "$resolution" \
    --batch-size "$batch_size" \
    --epochs "$epochs" \
    --target-mode 3-class \
    --loss ce-dice \
    --class-weights "${weights_array[@]}" \
    --lr 1e-4 \
    --num-workers 8
}

evaluate_candidate() {
  local name="$1"
  local resolution="$2"
  local split_dir="$3"
  local output_json="$4"
  local checkpoint="outputs/checkpoints/${name}/best_unet.pt"
  local split_csv="data/processed/${split_dir}/test_pairs.csv"

  if [ ! -f "$checkpoint" ]; then
    echo "Missing checkpoint for evaluation: $checkpoint" >&2
    return 1
  fi
  if [ ! -f "$split_csv" ]; then
    echo "Missing evaluation split CSV: $split_csv" >&2
    return 1
  fi

  python -m src.crisismap.evaluation.evaluate_unet \
    --root data/raw/xbd/train \
    --split-csv "$split_csv" \
    --checkpoint "$checkpoint" \
    --output "$output_json" \
    --image-size "$resolution" \
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

run_complete() {
  local history_json="$1"
  local expected_epochs="$2"
  local test_metrics_json="$3"

  history_complete "$history_json" "$expected_epochs" &&
    [ -f "$test_metrics_json" ]
}

run_candidate() {
  local candidate="$1"
  local name
  local resolution
  local split_dir
  local split_label
  local batch_size
  local weights
  local epochs
  local output_dir
  local checkpoint
  local history_json
  local test_metrics_json

  IFS="|" read -r name resolution split_dir split_label batch_size weights epochs <<< "$candidate"
  output_dir="outputs/checkpoints/${name}"
  checkpoint="${output_dir}/best_unet.pt"
  history_json="${output_dir}/metrics_history.json"
  test_metrics_json="outputs/predictions/${name}_test_metrics.json"

  echo
  echo "==> Candidate: $name"
  echo "resolution=${resolution} split=${split_label} batch_size=${batch_size} loss=ce-dice weights=${weights} lr=1e-4 epochs=${epochs}"
  echo "started=$(date)"

  if ! split_has_files "$split_dir"; then
    echo "Missing split files under data/processed/${split_dir}" >&2
    return 1
  fi

  if [ "$FORCE" = "1" ]; then
    echo "FORCE=1: removing previous output and metrics before fresh training."
    rm -rf -- "$output_dir"
    rm -f -- "$test_metrics_json"
  elif run_complete "$history_json" "$epochs" "$test_metrics_json"; then
    echo "Training history and test metrics JSON exist; skipping candidate."
    return 0
  elif [ -d "$output_dir" ]; then
    if history_complete "$history_json" "$epochs" && [ -f "$checkpoint" ]; then
      echo "Training history shows ${epochs} epochs and best checkpoint exists; evaluating test metrics."
      evaluate_candidate "$name" "$resolution" "$split_dir" "$test_metrics_json"
      return 0
    fi

    echo "WARNING: Candidate appears incomplete: $output_dir"
    echo "WARNING: Not evaluating best_unet.pt because it may come from partial training."
    echo "WARNING: Relaunch with:"
    echo "  sbatch --export=ALL,FORCE=1 slurm/train_unet_long_best_models_noleak.sbatch"
    return 0
  fi

  train_candidate "$name" "$resolution" "$split_dir" "$batch_size" "$weights" "$epochs"
  evaluate_candidate "$name" "$resolution" "$split_dir" "$test_metrics_json"
  echo "finished=$(date)"
}

ensure_required_files
create_noleak_splits_if_needed
ensure_candidate_splits

echo "No-leak long-training best U-Net models"
echo "FORCE=${FORCE}"
echo "Summary CSV: ${SUMMARY_CSV}"
echo "Evaluation protocol: each split's test_pairs.csv is the common full-data test copy"

for candidate in "${CANDIDATES[@]}"; do
  IFS="|" read -r name _resolution _split_dir _split_label _batch_size _weights _epochs <<< "$candidate"
  run_log="${HOME}/scratch/CrisisMap-AI/run_logs/${name}-${JOB_ID}.log"
  run_candidate "$candidate" 2>&1 | tee "$run_log"
done

python - <<'PY'
import csv
import json
from pathlib import Path

candidates = [
    ("unet_1024_ce_dice_w005_1_4_noleak_match_hist1000_bs2_250epochs", 1024, "noleak_match_hist1000", 2, "0.05 1.0 4.0", 250),
    ("unet_1024_ce_dice_w004_1_5_noleak_match_hist1000_bs3_250epochs", 1024, "noleak_match_hist1000", 3, "0.04 1.0 5.0", 250),
    ("unet_1024_ce_dice_w005_1_4_noleak_match_dmg001_bs3_250epochs", 1024, "noleak_match_dmg001", 3, "0.05 1.0 4.0", 250),
    ("unet_512_ce_dice_w005_1_4_noleak_old4_bs2_250epochs", 512, "noleak_old4", 2, "0.05 1.0 4.0", 250),
]


def load_metrics(path):
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def class_value(metrics, key, index):
    if not metrics:
        return None
    values = metrics.get(key, [])
    if not isinstance(values, list) or len(values) <= index:
        return None
    return values[index]


rows = []
for name, resolution, split_label, batch_size, weights, epochs in candidates:
    metrics_path = Path("outputs/predictions") / f"{name}_test_metrics.json"
    checkpoint_path = Path("outputs/checkpoints") / name / "best_unet.pt"
    metrics = load_metrics(metrics_path)

    if metrics is None:
        print(f"Skipping summary row; missing test metrics: {metrics_path}")
        continue

    rows.append({
        "experiment": name,
        "resolution": resolution,
        "split": split_label,
        "batch_size": batch_size,
        "loss": "ce-dice",
        "class_weights": weights,
        "lr": "1e-4",
        "epochs": epochs,
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

output = Path("outputs/predictions/unet_long_best_models_noleak_summary.csv")
output.parent.mkdir(parents=True, exist_ok=True)
fieldnames = [
    "experiment",
    "resolution",
    "split",
    "batch_size",
    "loss",
    "class_weights",
    "lr",
    "epochs",
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

print(f"Saved sorted summary CSV: {output}")
print("\nSorted by test damaged IoU, damaged F1, mean IoU:")
print(f"{'experiment':74s} {'res':>4s} {'split':>22s} {'bs':>3s} {'mIoU':>10s} {'iou_dmg':>10s} {'f1_dmg':>10s}")
for row in rows:
    print(
        f"{row['experiment']:74s} "
        f"{row['resolution']:>4d} "
        f"{row['split']:>22s} "
        f"{row['batch_size']:>3d} "
        f"{row['test_mean_iou'] or 0:10.4f} "
        f"{row['test_iou_damaged'] or 0:10.4f} "
        f"{row['test_f1_damaged'] or 0:10.4f}"
    )
PY
