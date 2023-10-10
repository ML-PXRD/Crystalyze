#!/bin/bash
#SBATCH -N 1                     # Request one node
#SBATCH -n 18                     # Request 4 CPU cores
#SBATCH --array=1                # Array job ID
#SBATCH --mem=45000              # Request 45GB memory
#SBATCH --time=96:00:00          # Set maximum wall time
#SBATCH --output="output/%j_output.txt"  # Set output file
#SBATCH --error="error/%j_error.txt"     # Set error file
#SBATCH --gres=gpu:volta:1     # Request 1 GPU (atm, code breaks over 2, see Understanding CDVAE doc)

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

# Pass use_cond_kld parameter to your run script
python cdvae/run.py data=$1 expname=$2 

#python scripts/evaluate.py --model_path /home/gridsan/tmackey/hydra/singlerun/2023-09-13/perov --tasks recon
#python scripts/compute_metrics.py --root_path /home/gridsan/tmackey/hydra/singlerun/2023-09-13/perov --tasks recon
