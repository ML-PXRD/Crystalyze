#!/bin/bash
#SBATCH -N 1                     # Request one node
#SBATCH -n 16                     # Request 4 CPU cores
#SBATCH --array=1                # Array job ID
#SBATCH --mem=45000              # Request 45GB memory
#SBATCH --time=96:00:00          # Set maximum wall time
#SBATCH --output="output/%j_output.txt"  # Set output file
#SBATCH --error="error/%j_error.txt"     # Set error file
#SBATCH --gres=gpu:volta:1     # Request 1 GPU

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
FORCE_ATOM_TYPES_FLAG=""
if [ "$3" == "True" ]; then
  FORCE_ATOM_TYPES_FLAG="--force_atom_types"
fi

FORCE_NUM_ATOMS_FLAG=""
if [ "$7" == "True" ]; then
  FORCE_NUM_ATOMS_FLAG="--force_num_atoms"
fi

echo "Second argument is $2"
# Initialize a counter
counter=0

# Run the loop $4 times
while [ $counter -lt $4 ]; do
    echo "Running iteration $((counter+1))"

  #check to see if $5 is empty
  if [ -z "$5" ]; then
    echo "\$5 is empty"
    python scripts/evaluate.py --model_path $1 --tasks recon --num_batches $2 $FORCE_NUM_ATOMS_FLAG $FORCE_ATOM_TYPES_FLAG --label $6
    #python scripts/compute_metrics.py --root_path $1 --tasks recon --compare_diffraction_patterns True
  else
    echo "\$5 is NOT empty"
    python scripts/evaluate.py --model_path $1 --tasks recon --num_batches $2 $FORCE_NUM_ATOMS_FLAG $FORCE_ATOM_TYPES_FLAG --test_set_override $5 --label $6
    #python scripts/compute_metrics.py --root_path $1 --tasks recon --compare_diffraction_patterns True --label $5
  fi

  # Increment the counter
  ((counter++))
done

#python scripts/evaluate.py --model_path /home/gridsan/tmackey/hydra/singlerun/2024-01-31/ae_pf/ --tasks recon --num_batches 1 --force_num_atoms --force_atom_types --test_set_override "unsolved_compounds" --label "wstoich_unsolved_compounds"
#python scripts/evaluate.py --model_path /home/gridsan/tmackey/hydra/singlerun/2024-01-29/augmented_vae_nopf --tasks recon --num_batches 1 --force_num_atoms --force_atom_types --test_set_override "PDF_unsolved_compounds" --label "wstoich_PDF_unsolved_compounds"
#python scripts/evaluate.py --model_path /home/gridsan/tmackey/hydra/singlerun/2024-01-29/augmented_vae_nopf --tasks recon --num_batches 1 --force_num_atoms --force_atom_types --test_set_override "Freedman_lab_full_subtraction" --label "Freedman_lab_full_subtraction"