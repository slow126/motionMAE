#!/bin/bash
#SBATCH --job-name=vf_phase0_smoke
#SBATCH --output=/home/slow1/Projects/MotionMAE/slurm_logs/perceiver_variable_flow/smoke/vf_phase0_smoke.log
#SBATCH --error=/home/slow1/Projects/MotionMAE/slurm_logs/perceiver_variable_flow/smoke/vf_phase0_smoke.log
#SBATCH --partition=cs
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64g

set -euo pipefail
cd /home/slow1/Projects/MotionMAE
source ~/.bashrc || true
conda activate cuda
srun python3 -u /home/slow1/Projects/MotionMAE/scripts/train_variable_flow_perceiver.py --config /home/slow1/Projects/MotionMAE/percieverIO_Exp/configs/phase0_smoke.yaml
