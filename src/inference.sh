#!/bin/bash
 
#SBATCH --job-name=deep_svdd_inference
#SBATCH --partition=student,shared,sharedp
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --exclude=destc0strapp03
#SBATCH --output=slurm-logs/inference/test1/output.log
#SBATCH --error=slurm-logs/inference/test1/error.log

# Load conda/mamba properly for SLURM
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate venv

nvidia-smi
python main.py hs_ds HSNet \
    ../log/deep_svdd_inference_SL1DB_50sub2pps_SkinPatch_100f_withillum \
    ../data \
    --load_model /home/denegasf/repo/negasa-fromsa-teshome-msc-thesis/src/Deep-SVDD-PyTorch/log/test1_training/model.tar \
    --pretrain False \
    --n_epochs 0 \