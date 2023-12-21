#!/bin/bash

#SBATCH --job-name=structure-processing
#SBATCH --output=job_%A_%a.out
#SBATCH --error=job_%A_%a.err
#SBATCH --array=0-40  # Adjust based on how many chunks you want
#SBATCH --time=02:00:00
#SBATCH --mem=45000              # Request 45GB memory
#SBATCH --cpus-per-task=1


echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"

#module add mpi/openmpi-4.1.3
#module load anaconda/2023a
#conda init bash
source /state/partition1/llgrid/pkg/anaconda/anaconda3-2023a/etc/profile.d/conda.sh
conda activate cdvae

python examiner.py 