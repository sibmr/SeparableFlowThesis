#!/bin/bash
#SBATCH --job-name="train_sepflow"
#SBATCH --partition=slurmqueue
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --output=checkpoints/debug/train_original4GPU/slurm_output/job_%j_output.txt
#SBATCH --error=checkpoints/debug/train_original4GPU/slurm_output/job_%j_error.txt

./checkpoints/debug/train_original4GPU/slurm/run_me.sh
