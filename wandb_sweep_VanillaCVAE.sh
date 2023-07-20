#!/bin/bash
#SBATCH --job-name=wandb_sweep_VanillaCVAE
#SBATCH --output=./output/logs/slurm-%j.log

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=20-0:00:00
#SBATCH --mem=20G

#SBATCH --ntasks-per-node=1       # 1 CPU core to drive GPU
#SBATCH --nodes=2        # 1 GPU node
#SBATCH --cpus-per-task=8
###SBATCH --cpus-per-gpu=10
#SBATCH --gres=gpu:1     # Request 1 GPU
###SBATCH --gres-flags=disable-binding
###SBATCH --partition=gpu
#SBATCH --nodelist=scgn01

echo 'activating virtual environment'
source ~/.bashrc
conda activate sharedenv

config_yaml='/mnt/mcfiles/hli/projects/cuan_2021_deeplearning-breastcancer/wandb_sweep_config_VanillaCVAE.yaml'
echo 'config:' $config_yaml

train_file='/mnt/mcfiles/hli/projects/cuan_2021_deeplearning-breastcancer/main.py'
echo 'train_file:' $train_file

project_name='cuan_2021_deeplearning-breastcancer'
echo 'project_name:' $project_name

echo 'running script'
python wandb_on_slurm.py $config_yaml $train_file $project_name

conda deactivate
