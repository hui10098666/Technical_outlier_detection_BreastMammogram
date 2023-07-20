#!/bin/bash
#SBATCH --job-name="analyze result"

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=12-0:00:00
#SBATCH --mem=20G

#SBATCH --ntasks-per-node=1       # 1 CPU core to drive GPU
#SBATCH --nodes=1        # 1 GPU node
#SBATCH --cpus-per-task=6
#SBATCH --partition=gpu

# Batch arrays
#SBATCH --array=0-3

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

echo 'activating virtual environment'
source ~/.bashrc
conda activate sharedenv

echo 'running script'
python analyze_result.py ${SLURM_ARRAY_TASK_ID}
