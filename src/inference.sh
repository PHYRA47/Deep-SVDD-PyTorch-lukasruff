#!/bin/bash
 
#SBATCH --job-name=deep_svdd_inference
#SBATCH --partition=student,shared,sharedp
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --exclude=destc0strapp03
#SBATCH --output=slurm-logs/output.log
#SBATCH --error=slurm-logs/error.log

nvidia-smi
python main.py hs_ds HSNet \
    ../log/prior \
    ../data \
    --load_model ../log/prior/model.tar \
    --pretrain False \
    --n_epochs 0