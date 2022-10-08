#!/bin/bash
#SBATCH --job-name="train_sepflow"
#SBATCH --partition=slurmqueue
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=2
#SBATCH --gres=gpu:4
#SBATCH --output=checkpoints/train_originalLongerChairs/slurm_output/job_%j_output.txt
#SBATCH --error=checkpoints/train_originalLongerChairs/slurm_output/job_%j_error.txt

./checkpoints/train_originalLongerChairs/slurm/run_me.sh
