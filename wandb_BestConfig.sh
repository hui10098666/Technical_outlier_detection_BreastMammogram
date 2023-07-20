#!/bin/bash
#SBATCH --job-name=wandb_BestConfig
#SBATCH --output=./output/logs/slurm-%j.log

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=5-0:00:00
#SBATCH --mem=20G

#SBATCH --ntasks-per-node=1       # 1 CPU core to drive GPU
#SBATCH --nodes=1        # 1 GPU node
#SBATCH --cpus-per-task=6
###SBATCH --cpus-per-gpu=10
#SBATCH --gres=gpu:1     # Request 1 GPU
###SBATCH --gres-flags=disable-binding
#SBATCH --partition=gpu

echo 'activating virtual environment'
source ~/.bashrc
conda activate sharedenv

echo 'running script'
python main.py

