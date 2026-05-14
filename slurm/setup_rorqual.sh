#!/usr/bin/env bash
set -euo pipefail

# Run from: ~/work/CrisisMap-AI

CODE_ROOT="${HOME}/work/CrisisMap-AI"
SCRATCH_ROOT="${HOME}/scratch/CrisisMap-AI"
DATA_ROOT="${SCRATCH_ROOT}/data"
OUTPUTS_ROOT="${SCRATCH_ROOT}/outputs"
LOGS_ROOT="${SCRATCH_ROOT}/logs"
RUN_LOGS_ROOT="${SCRATCH_ROOT}/run_logs"
TRITON_CACHE_ROOT="${SCRATCH_ROOT}/triton_cache"
VENV_ROOT="${HOME}/virtualenvs/crisismap-ai"

TRAIN_ARCHIVE="data/raw/archives/train_images_labels_targets.tar"
GEOTRANSFORMS_ARCHIVE="data/raw/archives/xview_geotransforms.json.tgz"
GEOTRANSFORMS_JSON="data/raw/geotransforms/xview_geotransforms.json"
INDEX_CSV="data/processed/xbd_train_index.csv"
SPLITS_FULL_DIR="data/processed/splits_full"

DISASTERS=(
  guatemala-volcano
  hurricane-florence
  hurricane-harvey
  hurricane-matthew
  hurricane-michael
  mexico-earthquake
  midwest-flooding
  palu-tsunami
  santa-rosa-wildfire
  socal-fire
)

step() {
  echo
  echo "==> $1"
}

require_file() {
  local path="$1"
  if [ ! -f "$path" ]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
}

ensure_symlink() {
  local link_path="$1"
  local target_path="$2"

  if [ -L "$link_path" ]; then
    if [ "$(readlink "$link_path")" = "$target_path" ]; then
      echo "Symlink already set: $link_path -> $target_path"
      return
    fi
    rm "$link_path"
  elif [ -e "$link_path" ]; then
    local backup="${link_path}.repo_placeholder.$(date +%Y%m%d%H%M%S)"
    echo "Moving existing repo path aside: $link_path -> $backup"
    mv "$link_path" "$backup"
  fi

  ln -s "$target_path" "$link_path"
  echo "Created symlink: $link_path -> $target_path"
}

dataset_ready() {
  [ -d data/raw/xbd/train/images ] &&
    [ -d data/raw/xbd/train/labels ] &&
    [ -d data/raw/xbd/train/targets ]
}

if [ "$(pwd)" != "$CODE_ROOT" ]; then
  echo "This script is intended to run from: $CODE_ROOT" >&2
  echo "Current directory: $(pwd)" >&2
  exit 1
fi

step "Load Alliance/Rorqual modules"
module --force purge
module load StdEnv/2023
module load python/3.11
module load gcc
module load arrow/23.0.1
module load cuda

step "Create scratch directories"
mkdir -p "${DATA_ROOT}/raw/archives"
mkdir -p "${DATA_ROOT}/raw/xbd"
mkdir -p "${DATA_ROOT}/raw/geotransforms"
mkdir -p "${DATA_ROOT}/processed"
mkdir -p "${OUTPUTS_ROOT}/checkpoints"
mkdir -p "${OUTPUTS_ROOT}/predictions"
mkdir -p "$LOGS_ROOT"
mkdir -p "$RUN_LOGS_ROOT"
mkdir -p "$TRITON_CACHE_ROOT"
mkdir -p "$(dirname "$VENV_ROOT")"

step "Create or refresh repository symlinks"
ensure_symlink data "${DATA_ROOT}"
ensure_symlink outputs "${OUTPUTS_ROOT}"

step "Create and activate Python virtual environment"
if [ ! -d "$VENV_ROOT" ]; then
  python -m venv "$VENV_ROOT"
fi
source "${VENV_ROOT}/bin/activate"
export TRITON_CACHE_DIR="$TRITON_CACHE_ROOT"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

step "Verify local dataset archives"
require_file "$TRAIN_ARCHIVE"
require_file "$GEOTRANSFORMS_ARCHIVE"

step "Extract xBD training archive if needed"
if dataset_ready; then
  echo "Dataset folders already exist; skipping xBD extraction."
else
  mkdir -p data/raw/xbd
  tar -xf "$TRAIN_ARCHIVE" -C data/raw/xbd
fi

step "Extract geotransforms archive if needed"
if [ -f "$GEOTRANSFORMS_JSON" ]; then
  echo "Geotransforms JSON already exists; skipping extraction."
else
  mkdir -p data/raw/geotransforms
  tar -xzf "$GEOTRANSFORMS_ARCHIVE" -C data/raw/geotransforms
  if [ ! -f "$GEOTRANSFORMS_JSON" ]; then
    found_json="$(find data/raw/geotransforms -name xview_geotransforms.json -type f | head -n 1 || true)"
    if [ -z "$found_json" ]; then
      echo "Could not find xview_geotransforms.json after extraction." >&2
      exit 1
    fi
    cp "$found_json" "$GEOTRANSFORMS_JSON"
  fi
fi

step "Verify extracted dataset folders"
test -d data/raw/xbd/train/images
test -d data/raw/xbd/train/labels
test -d data/raw/xbd/train/targets
require_file "$GEOTRANSFORMS_JSON"

step "Inspect xBD dataset"
python -m src.crisismap.data.inspect_xbd --root data/raw/xbd/train

step "Build xBD index"
mkdir -p data/processed
python -m src.crisismap.data.build_xbd_index \
  --root data/raw/xbd/train \
  --output "$INDEX_CSV"

step "Create full-data splits"
mkdir -p "$SPLITS_FULL_DIR"
python -m src.crisismap.data.create_xbd_splits \
  --index "$INDEX_CSV" \
  --output-dir "$SPLITS_FULL_DIR" \
  --disasters "${DISASTERS[@]}" \
  --val-size 0.15 \
  --test-size 0.15 \
  --min-nonzero-ratio 0.01 \
  --seed 42

step "Setup complete"
echo "Code root: $CODE_ROOT"
echo "Data root: $DATA_ROOT"
echo "Outputs root: $OUTPUTS_ROOT"
echo "Logs root: $LOGS_ROOT"
echo "Run logs root: $RUN_LOGS_ROOT"
echo "Triton cache: $TRITON_CACHE_DIR"
echo "Virtualenv: $VENV_ROOT"
echo "Index: $INDEX_CSV"
echo "Full splits: $SPLITS_FULL_DIR"
echo
echo "Useful next commands:"
echo "  bash slurm/check_cluster_status.sh"
echo "  sbatch slurm/smoke_unet_512.sbatch"
echo "  sbatch slurm/train_unet_full_1024.sbatch"
echo "  sbatch slurm/sweep_unet_1024.sbatch"
