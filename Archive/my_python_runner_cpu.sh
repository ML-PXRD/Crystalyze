#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --array=1
#SBATCH --mem=45000
#SBATCH --time=96:00:00
#SBATCH --output="output/%j_output.txt"
#SBATCH --error="error/%j_error.txt"

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"

module add mpi/openmpi-4.1.3

python test_python_script.py
