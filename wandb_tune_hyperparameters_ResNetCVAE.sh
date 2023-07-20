#!/bin/bash
#SBATCH --job-name="wandb tune hyperparameters ResNetCVAE"

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=8-0:00:00
#SBATCH --mem=20G

#SBATCH --ntasks-per-node=1       # 1 CPU core to drive GPU
#SBATCH --nodes=1        # 1 GPU node
#SBATCH --cpus-per-task=6
###SBATCH --cpus-per-gpu=10
#SBATCH --gres=gpu:1     # Request 1 GPU
###SBATCH --gres-flags=disable-binding
#SBATCH --partition=gpu

# Batch arrays
#SBATCH --array=9

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

echo 'activating virtual environment'
source ~/.bashrc
conda activate sharedenv

echo 'running script'
python wandb_tune_hyperparameters_ResNetCVAE.py ${SLURM_ARRAY_TASK_ID}
