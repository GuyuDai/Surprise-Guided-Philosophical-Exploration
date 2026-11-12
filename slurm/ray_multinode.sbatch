#!/usr/bin/env bash
#SBATCH --job-name=ai4ph-ray
#SBATCH --partition=h100
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100:4
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
mkdir -p logs

nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
head_ip="$(getent ahostsv4 "${nodes[0]}" | awk 'NR == 1 {print $1}')"
[[ -n "$head_ip" ]] || { echo "could not resolve Ray head IPv4 address" >&2; exit 1; }

srun scripts/container_exec.sh ray symmetric-run \
    --address "$head_ip:6379" \
    --min-nodes "$SLURM_NNODES" \
    --num-cpus "$SLURM_CPUS_PER_TASK" \
    --num-gpus 4 \
    -- python scripts/ray_entrypoint.py

