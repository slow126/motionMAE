#!/bin/bash
#SBATCH --job-name=vf_phase3_full
#SBATCH --output=/home/slow1/Projects/motionMAE/snapshots/perceiver_variable_flow/_slurm/full/vf_phase3_full-%j.log
#SBATCH --error=/home/slow1/Projects/motionMAE/snapshots/perceiver_variable_flow/_slurm/full/vf_phase3_full-%j.log
#SBATCH --partition=cs
#SBATCH --time=23:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64g
#SBATCH --qos=cs

set -euo pipefail
cd /home/slow1/Projects/motionMAE
source ~/.bashrc || true
conda activate cuda
srun python3 -u /home/slow1/Projects/motionMAE/scripts/train_variable_flow_perceiver.py --config /home/slow1/Projects/motionMAE/percieverIO_Exp/configs/phase3_rc.yaml
