#!/bin/bash
#SBATCH --job-name="train_sepflow"
#SBATCH --partition=slurmqueue
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=checkpoints/debug/train_original1GPU/slurm_output/job_%j_output.txt
#SBATCH --error=checkpoints/debug/train_original1GPU/slurm_output/job_%j_error.txt

./checkpoints/debug/train_original1GPU/slurm/run_me.sh
