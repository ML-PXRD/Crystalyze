#!/bin/bash
#SBATCH -N 2                    # Request one node
#SBATCH -n 80                     # Request 4 CPU cores
#SBATCH --array=1                # Array job ID
#SBATCH --mem=45000              # Request 45GB memory
#SBATCH --time=96:00:00          # Set maximum wall time

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

python create_diffraction_csv.py True False /home/gridsan/tmackey/materials_discovery/data/gnome_data/

#python scripts/evaluate.py --model_path /home/gridsan/tmackey/hydra/singlerun/2023-09-13/perov --tasks recon
#python scripts/compute_metrics.py --root_path /home/gridsan/tmackey/hydra/singlerun/2023-09-13/perov --tasks recon
