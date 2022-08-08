#!/bin/bash
#SBATCH --job-name="train_sepflow"
#SBATCH --partition=slurmqueue
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=checkpoints/train_memsave/slurm_output/job_%j_output.txt
#SBATCH --error=checkpoints/train_memsave/slurm_output/job_%j_error.txt

./checkpoints/train_memsave/slurm/run_me.sh
