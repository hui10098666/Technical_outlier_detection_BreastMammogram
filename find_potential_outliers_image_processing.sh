#!/bin/bash
#SBATCH --job-name="find potential outliers image processing"

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=3-0:00:00
#SBATCH --mem=20G

#SBATCH --ntasks-per-node=1       # 1 CPU core to drive GPU
#SBATCH --nodes=1        # 1 GPU node
#SBATCH --cpus-per-task=6
#SBATCH --partition=gpu

# Batch arrays
#SBATCH --array=0-23

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

echo 'activating virtual environment'
source ~/.bashrc
conda activate sharedenv

echo 'running script'
python find_potential_outliers_image_processing_args.py ${SLURM_ARRAY_TASK_ID}
