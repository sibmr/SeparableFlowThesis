#!/bin/bash
#SBATCH --job-name="train_sepflow"
#SBATCH --partition=slurmqueue
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --output=checkpoints/debug/train_original2GPU/slurm_output/job_%j_output.txt
#SBATCH --error=checkpoints/debug/train_original2GPU/slurm_output/job_%j_error.txt

./checkpoints/debug/train_original2GPU/slurm/run_me.sh
