#!/bin/bash
#SBATCH --job-name="train_sepflow"
#SBATCH --partition=slurmqueue
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/home/bihlmasn/git/thesis/SeparableFlow/slurm_output/job_%j_output.txt
#SBATCH --error=/home/bihlmasn/git/thesis/SeparableFlow/slurm_output/job_%j_error.txt

/home/bihlmasn/git/thesis/SeparableFlow/scripts/slurm/run_me.sh
