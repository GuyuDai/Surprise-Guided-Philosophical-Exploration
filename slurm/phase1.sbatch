#!/usr/bin/env bash
#SBATCH --job-name=ai4ph-phase1
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
mkdir -p logs
export HF_HUB_OFFLINE=1
scripts/container_exec.sh python -m philo_mcts.cli --config configs/phase1.json

