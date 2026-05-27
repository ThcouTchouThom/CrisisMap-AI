#!/usr/bin/env bash

set -euo pipefail
cd "${HOME}/work/CrisisMap-AI"

echo "Submitting Building100 resume/relaunch jobs."
echo "RESUME_INCOMPLETE=1 resumes local U-Net partial last_building.pt checkpoints."
echo "Rows without checkpoints train normally; complete rows are skipped by the runner."
echo

CONFIG_CSV="${CONFIG_CSV:-configs/building100_sweep_v1_relaunch.csv}" \
RESUME_INCOMPLETE="${RESUME_INCOMPLETE:-1}" \
WAIT_FOR_LONG250="${WAIT_FOR_LONG250:-0}" \
bash slurm/submit_building100_sweep_v1.sh
