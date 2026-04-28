#!/bin/bash
#SBATCH --job-name=vf_phase1_full
#SBATCH --output=/home/spencer/Projects/MotionMAE/slurm_logs/perceiver_variable_flow/full/vf_phase1_full.log
#SBATCH --error=/home/spencer/Projects/MotionMAE/slurm_logs/perceiver_variable_flow/full/vf_phase1_full.log
#SBATCH --partition=cs
#SBATCH --time=23:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64g

set -euo pipefail
cd /home/spencer/Projects/MotionMAE
source ~/.bashrc || true
conda activate cuda
srun python3 -u /home/spencer/Projects/MotionMAE/scripts/train_variable_flow_perceiver.py --config /home/spencer/Projects/MotionMAE/percieverIO_Exp/configs/phase1_rc.yaml
