#!/usr/bin/env bash

set -euo pipefail
cd "${HOME}/work/CrisisMap-AI"

echo "Submitting damage extra resume/relaunch jobs."
echo "RESUME_INCOMPLETE=1 resumes partial last_unet.pt checkpoints."
echo "Complete runs are skipped by slurm/run_damage_extra_config.sh."
echo

CONFIG_CSV="${CONFIG_CSV:-configs/damage_extra_sweep_v1_resume.csv}" \
RESUME_INCOMPLETE="${RESUME_INCOMPLETE:-1}" \
bash slurm/submit_damage_extra_sweep_v1.sh
