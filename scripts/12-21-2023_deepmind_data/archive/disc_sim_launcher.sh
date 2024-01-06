#!/bin/bash

#SBATCH --job-name=graph_preloading
#SBATCH --output=output/log_%A_%a.out
#SBATCH --error=error/log_%A_%a.err
#SBATCH --array=0-99
#SBATCH --time=02:00:00
#SBATCH --mem=45000
#SBATCH --cpus-per-task=4

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"

#module add mpi/openmpi-4.1.3
#module load anaconda/2023a
#conda init bash
source /state/partition1/llgrid/pkg/anaconda/anaconda3-2023a/etc/profile.d/conda.sh
conda activate cdvae

#python test_python_script.py
echo "Name of run is $2 "
echo "Data used is $1" 
python disc_sim_xrd.py $SLURM_ARRAY_TASK_ID 