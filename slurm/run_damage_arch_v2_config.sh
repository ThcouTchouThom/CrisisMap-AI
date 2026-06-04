#!/usr/bin/env bash
#SBATCH --job-name=damage_arch_v2
#SBATCH --account=def-zonata_gpu
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --output=/home/tgrjlt2/scratch/CrisisMap-AI/logs/%x-%j.out
#SBATCH --error=/home/tgrjlt2/scratch/CrisisMap-AI/logs/%x-%j.err
#SBATCH --mail-user=t.gourjault@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

# Email notifications to avoid frequent scheduler polling.
# Thin v2 wrapper around the generic Axis 2 damage architecture runner.

set -euo pipefail
cd "${HOME}/work/CrisisMap-AI"
export PYTHONPATH="${PWD}/src:${PWD}:${PYTHONPATH:-}"

export CONFIG_CSV="${CONFIG_CSV:-configs/damage_arch_sweep_v2.csv}"
bash slurm/run_damage_arch_config.sh
