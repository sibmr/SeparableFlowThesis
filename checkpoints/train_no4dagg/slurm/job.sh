#!/bin/bash
#SBATCH --job-name="train_sepflow"
#SBATCH --partition=slurmqueue
#SBATCH --mem=80gb 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --output=checkpoints/train_no4dagg/slurm_output/job_%j_output.txt
#SBATCH --error=checkpoints/train_no4dagg/slurm_output/job_%j_error.txt

./checkpoints/train_no4dagg/slurm/run_me.sh
