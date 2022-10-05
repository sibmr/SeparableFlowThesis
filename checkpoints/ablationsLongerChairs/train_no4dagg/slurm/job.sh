#!/bin/bash
#SBATCH --job-name="train_sepflow"
#SBATCH --partition=slurmqueue
#SBATCH --mem=80gb 
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=2
#SBATCH --gres=gpu:4
#SBATCH --output=checkpoints/ablationsLongerChairs/train_no4dagg/slurm_output/job_%j_output.txt
#SBATCH --error=checkpoints/ablationsLongerChairs/train_no4dagg/slurm_output/job_%j_error.txt

./checkpoints/ablationsLongerChairs/train_no4dagg/slurm/run_me.sh
