#!/bin/bash

#SBATCH --job-name=graph_preloading
#SBATCH --output=output/log_%A_%a.out
#SBATCH --error=error/log_%A_%a.err
#SBATCH --array=0-99
#SBATCH --time=05:00:00
#SBATCH --mem=45000
#SBATCH --cpus-per-task=1

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"

#module add mpi/openmpi-4.1.3
#module load anaconda/2023a
#conda init bash
source /state/partition1/llgrid/pkg/anaconda/anaconda3-2023a/etc/profile.d/conda.sh
conda activate cdvae

#python test_python_script.py
echo "Name of run is $1"
echo "Data used is $2"

#check if $1 is "graph"
if [ $1 == "graph" ]
then
    python graph_preloading_data.py $SLURM_ARRAY_TASK_ID 100
elif [ $1 == "diffraction" ]
then
    python xrd_prep.py $SLURM_ARRAY_TASK_ID $2
elif [ $1 == "disc_sim_xrd" ]
then
    python disc_sim_xrd_prep.py $SLURM_ARRAY_TASK_ID $2
elif [ $1 == "pv_xrd" ] 
then 
    python pv_xrd_prep.py --n_workers 100 --worker_num $SLURM_ARRAY_TASK_ID
fi
