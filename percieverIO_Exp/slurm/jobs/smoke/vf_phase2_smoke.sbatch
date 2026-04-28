#!/bin/bash
#SBATCH --job-name=vf_phase2_smoke
#SBATCH --output=/home/spencer/Projects/MotionMAE/slurm_logs/perceiver_variable_flow/smoke/vf_phase2_smoke.log
#SBATCH --error=/home/spencer/Projects/MotionMAE/slurm_logs/perceiver_variable_flow/smoke/vf_phase2_smoke.log
#SBATCH --partition=cs
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64g

set -euo pipefail
cd /home/spencer/Projects/MotionMAE
source ~/.bashrc || true
conda activate cuda
srun python3 -u /home/spencer/Projects/scripts/train_variable_flow_perceiver.py --config /home/spencer/Projects/MotionMAE/configs/phase2_smoke.yaml
