#!/bin/bash
#SBATCH -N 1                     # Request one node
#SBATCH -n 16                     # Request 4 CPU cores
#SBATCH --array=1                # Array job ID
#SBATCH --mem=45000              # Request 45GB memory
#SBATCH --time=96:00:00          # Set maximum wall time
#SBATCH --output="output/%j_output.txt"  # Set output file
#SBATCH --error="error/%j_error.txt"     # Set error file
#SBATCH --gres=gpu:volta:2     # Request 1 GPU

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"

#module add mpi/openmpi-4.1.3
#module load anaconda/2023a
#conda init bash
source /state/partition1/llgrid/pkg/anaconda/anaconda3-2023a/etc/profile.d/conda.sh
conda activate cdvae
#python test_python_script.py
#python cdvae/run.py data=perov expname=perov
echo "Second argument is $2"
python scripts/evaluate.py --model_path $1 --tasks recon --force_num_atoms --num_batches $2 --save_traj True
python scripts/compute_metrics.py --root_path $1 --tasks recon
