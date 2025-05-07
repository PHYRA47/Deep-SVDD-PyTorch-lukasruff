#!/bin/bash
 
#SBATCH --job-name=deep_svdd_training
#SBATCH --partition=student,shared,sharedp
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --exclude=destc0strapp03
#SBATCH --output=slurm-logs/training/test1/output.log
#SBATCH --error=slurm-logs/training/test1/error.log

# Load conda/mamba properly for SLURM
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate venv

# Print GPU information
nvidia-smi

# Run the training script
python main.py hs_ds HSNet ../log/test1 ../data --n_epochs 4 --batch_size 32 --ae_n_epochs 10 --ae_batch_size 32