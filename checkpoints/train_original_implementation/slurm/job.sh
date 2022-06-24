#!/bin/bash
#SBATCH --job-name="train_sepflow"
#SBATCH --partition=slurmqueue
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --output=checkpoints/train_original_implementation/slurm_output/job_%j_output.txt
#SBATCH --error=checkpoints/train_original_implementation/slurm_output/job_%j_error.txt

./checkpoints/train_original_implementation/slurm/run_me.sh
