#!/bin/bash
#SBATCH -N 1                     # Request one node
#SBATCH -n 16                     # Request 4 CPU cores
#SBATCH --array=1                # Array job ID
#SBATCH --mem=45000              # Request 45GB memory
#SBATCH --time=96:00:00          # Set maximum wall time
#SBATCH --output="output/%j_output.txt"  # Set output file
#SBATCH --error="error/%j_error.txt"     # Set error file
#SBATCH --gres=gpu:volta:1     # Request 1 GPU

# module load mpi/openmpi-4.1.5
# module load nccl/2.18.1-cuda11.8
source /state/partition1/llgrid/pkg/anaconda/anaconda3-2023a/etc/profile.d/conda.sh
conda activate cdvae

python snapper.py