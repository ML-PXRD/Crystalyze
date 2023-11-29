#!/bin/bash
#SBATCH -N 1                     # Request one node
#SBATCH -n 16                     # Request 4 CPU cores
#SBATCH --array=1                # Array job ID
#SBATCH --mem=45000              # Request 45GB memory
#SBATCH --time=96:00:00          # Set maximum wall time
#SBATCH --gres=gpu:volta:1     # Request 1 GPU

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"

#module add mpi/openmpi-4.1.3
#module load anaconda/2023a
module load cuda/11.8
#conda init bash
#source /state/partition1/llgrid/pkg/anaconda/anaconda3-2023a/etc/profile.d/conda.sh - - TEMPORARILY OFF FOR PEROV CFTCP
#source /state/partition1/llgrid/pkg/anaconda/anaconda3-2021a/etc/profile.d/conda.sh
#conda activate FTCP_env - TEMPORARILY OFF FOR PEROV CFTCP

python "lattice_parameters_convet_model.py"
