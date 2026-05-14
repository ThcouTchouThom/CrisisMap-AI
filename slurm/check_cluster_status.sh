#!/usr/bin/env bash
set -u

CODE_ROOT="${HOME}/work/CrisisMap-AI"
SCRATCH_ROOT="${HOME}/scratch/CrisisMap-AI"
VENV_ROOT="${HOME}/virtualenvs/crisismap-ai"

echo "Aftermath / CrisisMap AI Rorqual status"
echo "Current directory: $(pwd)"
echo "Expected code root: $CODE_ROOT"
echo "Hostname: $(hostname)"
echo

echo "Git:"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "  Branch: $(git rev-parse --abbrev-ref HEAD)"
  echo "  Commit: $(git log -1 --oneline)"
else
  echo "  Not inside a git repository."
fi
echo

echo "Symlinks:"
if [ -L data ]; then
  echo "  data -> $(readlink data)"
else
  echo "  data symlink missing"
fi
if [ -L outputs ]; then
  echo "  outputs -> $(readlink outputs)"
else
  echo "  outputs symlink missing"
fi
echo

echo "Scratch directories:"
for path in \
  "${SCRATCH_ROOT}/data" \
  "${SCRATCH_ROOT}/outputs" \
  "${SCRATCH_ROOT}/logs" \
  "${SCRATCH_ROOT}/run_logs" \
  "${SCRATCH_ROOT}/triton_cache"
do
  if [ -d "$path" ]; then
    echo "  OK: $path"
  else
    echo "  MISSING: $path"
  fi
done
echo

echo "Data files:"
for path in \
  data/raw/archives/train_images_labels_targets.tar \
  data/raw/archives/xview_geotransforms.json.tgz \
  data/raw/xbd/train \
  data/processed/splits_full/train_pairs.csv
do
  if [ -e "$path" ]; then
    echo "  OK: $path"
  else
    echo "  MISSING: $path"
  fi
done
echo

PYTHON="${VENV_ROOT}/bin/python"
if [ -x "$PYTHON" ]; then
  echo "Python:"
  "$PYTHON" --version
  "$PYTHON" - <<'PY'
try:
    import torch
    print(f"  torch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
except Exception as exc:
    print(f"  Could not inspect torch/CUDA: {exc}")
PY
else
  echo "Python: ${PYTHON} not found or not executable"
fi
