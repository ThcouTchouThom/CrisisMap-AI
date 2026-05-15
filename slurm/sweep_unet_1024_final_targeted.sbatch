#!/usr/bin/env bash
#SBATCH --job-name=aftermath_final1024
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

# Final targeted 1024 U-Net sweep around the best matched split.
# To force reruns: sbatch --export=ALL,FORCE=1 slurm/sweep_unet_1024_final_targeted.sbatch
# To override the 512 champion checkpoint:
#   sbatch --export=ALL,CHAMPION_512_CHECKPOINT=outputs/checkpoints/<name>/best_unet.pt slurm/sweep_unet_1024_final_targeted.sbatch

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
SUMMARY_CSV="outputs/predictions/unet_1024_final_targeted_summary.csv"
FULL_TEST_CSV="data/processed/splits_full/test_pairs.csv"
CHAMPION_512_CHECKPOINT="${CHAMPION_512_CHECKPOINT:-outputs/checkpoints/unet_512_ce_dice_w005_1_4_50epochs/best_unet.pt}"
CHAMPION_512_METRICS="outputs/predictions/champion_512_full_test_recheck_metrics.json"
export CHAMPION_512_CHECKPOINT
export CHAMPION_512_METRICS

CANDIDATES=(
  "unet_1024_ce_dice_w005_1_4_match_hist1000_bs3_50epochs|splits_match_old_hist1000|match_hist1000|3|0.05 1.0 4.0|50"
  "unet_1024_ce_dice_w005_1_4_match_hist1000_bs4_50epochs|splits_match_old_hist1000|match_hist1000|4|0.05 1.0 4.0|50"
  "unet_1024_ce_dice_w004_1_5_match_hist1000_bs2_50epochs|splits_match_old_hist1000|match_hist1000|2|0.04 1.0 5.0|50"
  "unet_1024_ce_dice_w004_1_5_match_hist1000_bs3_50epochs|splits_match_old_hist1000|match_hist1000|3|0.04 1.0 5.0|50"
  "unet_1024_ce_dice_w003_1_6_match_hist1000_bs2_50epochs|splits_match_old_hist1000|match_hist1000|2|0.03 1.0 6.0|50"
  "unet_1024_ce_dice_w003_1_6_match_hist1000_bs3_50epochs|splits_match_old_hist1000|match_hist1000|3|0.03 1.0 6.0|50"
  "unet_1024_ce_dice_w005_1_4_match_dmg001_bs3_50epochs|splits_match_old_dmg001|match_dmg001|3|0.05 1.0 4.0|50"
  "unet_1024_ce_dice_w005_1_4_match_hist1000_bs2_100epochs|splits_match_old_hist1000|match_hist1000|2|0.05 1.0 4.0|100"
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

  if [ ! -f "$FULL_TEST_CSV" ]; then
    echo "Missing full-data test split: $FULL_TEST_CSV" >&2
    echo "Run: bash slurm/setup_rorqual.sh" >&2
    return 1
  fi

  if [ ! -f "$CHAMPION_512_CHECKPOINT" ]; then
    echo "Missing champion 512 checkpoint: $CHAMPION_512_CHECKPOINT" >&2
    echo "Override with:" >&2
    echo "  sbatch --export=ALL,CHAMPION_512_CHECKPOINT=outputs/checkpoints/<name>/best_unet.pt slurm/sweep_unet_1024_final_targeted.sbatch" >&2
    return 1
  fi
}

create_matched_splits_if_needed() {
  if split_has_files "splits_match_old_dmg001" &&
     split_has_files "splits_match_old_hist1000"; then
    return 0
  fi

  echo "Matched split folders are missing; creating them from xbd_train_index.csv."
  python scripts/create_matched_damage_splits.py \
    --index data/processed/xbd_train_index.csv \
    --output-root data/processed \
    --seed 42 \
    --val-size 0.15 \
    --test-size 0.15
}

evaluate_champion_512() {
  if [ "$FORCE" = "1" ]; then
    rm -f -- "$CHAMPION_512_METRICS"
  fi

  if [ -f "$CHAMPION_512_METRICS" ]; then
    echo "Champion 512 full-test metrics already exist; skipping recheck: $CHAMPION_512_METRICS"
    return 0
  fi

  echo "Re-evaluating champion 512 checkpoint on full-data test split."
  echo "checkpoint=${CHAMPION_512_CHECKPOINT}"
  python -m src.crisismap.evaluation.evaluate_unet \
    --root data/raw/xbd/train \
    --split-csv "$FULL_TEST_CSV" \
    --checkpoint "$CHAMPION_512_CHECKPOINT" \
    --output "$CHAMPION_512_METRICS" \
    --image-size 512 \
    --batch-size 2 \
    --target-mode 3-class \
    --num-workers 8
}

train_candidate() {
  local name="$1"
  local split_dir="$2"
  local batch_size="$3"
  local weights="$4"
  local epochs="$5"
  local output_dir="outputs/checkpoints/${name}"
  local weights_array

  read -r -a weights_array <<< "$weights"

  python -m src.crisismap.training.train_unet \
    --root data/raw/xbd/train \
    --train-csv "data/processed/${split_dir}/train_pairs.csv" \
    --val-csv "data/processed/${split_dir}/val_pairs.csv" \
    --output-dir "$output_dir" \
    --image-size 1024 \
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
  local split_csv="$2"
  local output_json="$3"
  local checkpoint="outputs/checkpoints/${name}/best_unet.pt"

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

run_complete() {
  local history_json="$1"
  local expected_epochs="$2"
  local own_metrics_json="$3"
  local full_metrics_json="$4"

  history_complete "$history_json" "$expected_epochs" &&
    [ -f "$own_metrics_json" ] &&
    [ -f "$full_metrics_json" ]
}

evaluate_missing_metrics() {
  local name="$1"
  local split_dir="$2"
  local own_metrics_json="outputs/predictions/${name}_own_test_metrics.json"
  local full_metrics_json="outputs/predictions/${name}_full_test_metrics.json"

  if [ ! -f "$own_metrics_json" ]; then
    evaluate_candidate "$name" "data/processed/${split_dir}/test_pairs.csv" "$own_metrics_json"
  fi
  if [ ! -f "$full_metrics_json" ]; then
    evaluate_candidate "$name" "$FULL_TEST_CSV" "$full_metrics_json"
  fi
}

run_candidate() {
  local candidate="$1"
  local name
  local split_dir
  local split_label
  local batch_size
  local weights
  local epochs
  local output_dir
  local checkpoint
  local history_json
  local own_metrics_json
  local full_metrics_json

  IFS="|" read -r name split_dir split_label batch_size weights epochs <<< "$candidate"
  output_dir="outputs/checkpoints/${name}"
  checkpoint="${output_dir}/best_unet.pt"
  history_json="${output_dir}/metrics_history.json"
  own_metrics_json="outputs/predictions/${name}_own_test_metrics.json"
  full_metrics_json="outputs/predictions/${name}_full_test_metrics.json"

  echo
  echo "==> Candidate: $name"
  echo "split=${split_label} batch_size=${batch_size} loss=ce-dice weights=${weights} lr=1e-4 epochs=${epochs}"
  echo "started=$(date)"

  if ! split_has_files "$split_dir"; then
    echo "Missing split files under data/processed/${split_dir}" >&2
    return 1
  fi

  if [ "$FORCE" = "1" ]; then
    echo "FORCE=1: removing previous output and metrics before fresh training."
    rm -rf -- "$output_dir"
    rm -f -- "$own_metrics_json" "$full_metrics_json"
  elif run_complete "$history_json" "$epochs" "$own_metrics_json" "$full_metrics_json"; then
    echo "Training history and both metrics JSONs exist; skipping candidate."
    return 0
  elif [ -d "$output_dir" ]; then
    if history_complete "$history_json" "$epochs" && [ -f "$checkpoint" ]; then
      echo "Training history shows ${epochs} epochs and best checkpoint exists; evaluating missing metrics only."
      evaluate_missing_metrics "$name" "$split_dir"
      return 0
    fi

    echo "WARNING: Candidate appears incomplete: $output_dir"
    echo "WARNING: Not evaluating best_unet.pt because it may come from partial training."
    echo "WARNING: Relaunch with:"
    echo "  sbatch --export=ALL,FORCE=1 slurm/sweep_unet_1024_final_targeted.sbatch"
    return 0
  fi

  train_candidate "$name" "$split_dir" "$batch_size" "$weights" "$epochs"
  evaluate_missing_metrics "$name" "$split_dir"
  echo "finished=$(date)"
}

ensure_required_files
create_matched_splits_if_needed
evaluate_champion_512

echo "Final targeted 1024 sweep"
echo "FORCE=${FORCE}"
echo "Summary CSV: ${SUMMARY_CSV}"
echo "Fair comparison test split: ${FULL_TEST_CSV}"

for candidate in "${CANDIDATES[@]}"; do
  IFS="|" read -r name _split_dir _split_label _batch_size _weights _epochs <<< "$candidate"
  run_log="${HOME}/scratch/CrisisMap-AI/run_logs/${name}-${JOB_ID}.log"
  run_candidate "$candidate" 2>&1 | tee "$run_log"
done

python - <<'PY'
import csv
import json
import os
from pathlib import Path

champion_checkpoint = os.environ.get(
    "CHAMPION_512_CHECKPOINT",
    "outputs/checkpoints/unet_512_ce_dice_w005_1_4_50epochs/best_unet.pt",
)
champion_metrics_path = Path(
    os.environ.get(
        "CHAMPION_512_METRICS",
        "outputs/predictions/champion_512_full_test_recheck_metrics.json",
    )
)

candidates = [
    ("unet_1024_ce_dice_w005_1_4_match_hist1000_bs3_50epochs", 1024, "match_hist1000", 3, "0.05 1.0 4.0", 50),
    ("unet_1024_ce_dice_w005_1_4_match_hist1000_bs4_50epochs", 1024, "match_hist1000", 4, "0.05 1.0 4.0", 50),
    ("unet_1024_ce_dice_w004_1_5_match_hist1000_bs2_50epochs", 1024, "match_hist1000", 2, "0.04 1.0 5.0", 50),
    ("unet_1024_ce_dice_w004_1_5_match_hist1000_bs3_50epochs", 1024, "match_hist1000", 3, "0.04 1.0 5.0", 50),
    ("unet_1024_ce_dice_w003_1_6_match_hist1000_bs2_50epochs", 1024, "match_hist1000", 2, "0.03 1.0 6.0", 50),
    ("unet_1024_ce_dice_w003_1_6_match_hist1000_bs3_50epochs", 1024, "match_hist1000", 3, "0.03 1.0 6.0", 50),
    ("unet_1024_ce_dice_w005_1_4_match_dmg001_bs3_50epochs", 1024, "match_dmg001", 3, "0.05 1.0 4.0", 50),
    ("unet_1024_ce_dice_w005_1_4_match_hist1000_bs2_100epochs", 1024, "match_hist1000", 2, "0.05 1.0 4.0", 100),
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


def make_row(
    experiment,
    resolution,
    split,
    batch_size,
    class_weights,
    epochs,
    checkpoint,
    own_metrics_path,
    full_metrics_path,
):
    own = load_metrics(own_metrics_path)
    full = load_metrics(full_metrics_path)
    if full is None:
        print(f"Skipping summary row; missing full-test metrics: {full_metrics_path}")
        return None

    return {
        "experiment": experiment,
        "resolution": resolution,
        "split": split,
        "batch_size": batch_size,
        "loss": "ce-dice",
        "class_weights": class_weights,
        "lr": "1e-4",
        "epochs": epochs,
        "own_pixel_accuracy": own.get("pixel_accuracy") if own else None,
        "own_mean_iou": own.get("mean_iou") if own else None,
        "own_iou_damaged": class_value(own, "iou_per_class", 2),
        "own_f1_damaged": class_value(own, "f1_per_class", 2),
        "full_pixel_accuracy": full.get("pixel_accuracy"),
        "full_mean_iou": full.get("mean_iou"),
        "full_iou_background": class_value(full, "iou_per_class", 0),
        "full_iou_no_damage": class_value(full, "iou_per_class", 1),
        "full_iou_damaged": class_value(full, "iou_per_class", 2),
        "full_precision_damaged": class_value(full, "precision_per_class", 2),
        "full_recall_damaged": class_value(full, "recall_per_class", 2),
        "full_f1_damaged": class_value(full, "f1_per_class", 2),
        "checkpoint": str(checkpoint),
        "own_metrics_json": str(own_metrics_path),
        "full_metrics_json": str(full_metrics_path),
    }


rows = []
champion = make_row(
    experiment="champion_512_full_test_recheck",
    resolution=512,
    split="splits_full",
    batch_size=2,
    class_weights="0.05 1.0 4.0",
    epochs=50,
    checkpoint=champion_checkpoint,
    own_metrics_path=champion_metrics_path,
    full_metrics_path=champion_metrics_path,
)
if champion:
    rows.append(champion)

for name, resolution, split_label, batch_size, weights, epochs in candidates:
    own_metrics_path = Path("outputs/predictions") / f"{name}_own_test_metrics.json"
    full_metrics_path = Path("outputs/predictions") / f"{name}_full_test_metrics.json"
    checkpoint_path = Path("outputs/checkpoints") / name / "best_unet.pt"
    row = make_row(
        experiment=name,
        resolution=resolution,
        split=split_label,
        batch_size=batch_size,
        class_weights=weights,
        epochs=epochs,
        checkpoint=checkpoint_path,
        own_metrics_path=own_metrics_path,
        full_metrics_path=full_metrics_path,
    )
    if row:
        rows.append(row)

rows.sort(
    key=lambda row: (
        row["full_iou_damaged"] if row["full_iou_damaged"] is not None else -1,
        row["full_f1_damaged"] if row["full_f1_damaged"] is not None else -1,
        row["full_mean_iou"] if row["full_mean_iou"] is not None else -1,
    ),
    reverse=True,
)

output = Path("outputs/predictions/unet_1024_final_targeted_summary.csv")
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
    "own_pixel_accuracy",
    "own_mean_iou",
    "own_iou_damaged",
    "own_f1_damaged",
    "full_pixel_accuracy",
    "full_mean_iou",
    "full_iou_background",
    "full_iou_no_damage",
    "full_iou_damaged",
    "full_precision_damaged",
    "full_recall_damaged",
    "full_f1_damaged",
    "checkpoint",
    "own_metrics_json",
    "full_metrics_json",
]
with output.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved sorted summary CSV: {output}")
print("\nSorted by full-test damaged IoU, damaged F1, mean IoU:")
print(f"{'experiment':68s} {'res':>4s} {'split':>14s} {'bs':>3s} {'full_mIoU':>10s} {'full_iou_dmg':>12s} {'full_f1_dmg':>11s}")
for row in rows:
    print(
        f"{row['experiment']:68s} "
        f"{row['resolution']:>4d} "
        f"{row['split']:>14s} "
        f"{row['batch_size']:>3d} "
        f"{row['full_mean_iou'] or 0:10.4f} "
        f"{row['full_iou_damaged'] or 0:12.4f} "
        f"{row['full_f1_damaged'] or 0:11.4f}"
    )
PY
