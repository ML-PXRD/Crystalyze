#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:2
#SBATCH --ntasks=18
#SBATCH --mem=45000
#SBATCH --time=96:00:00
##SBATCH --exclusive

ps -elf | grep python
echo "Node: $(hostname)"

export MASTER_PORT=$(shuf -i 49152-65535 -n 1)
# module load mpi/openmpi-4.1.5
# module load nccl/2.18.1-cuda11.8
source /state/partition1/llgrid/pkg/anaconda/anaconda3-2023a/etc/profile.d/conda.sh
conda activate cdvae

# Pass the max_num_atoms parameter to the Hydra configuration
python cdvae/run.py data=$1 expname=$2 max_num_atoms=$3

# readarray -t nodes <<< $(scontrol show hostnames $SLURM_NODELIST)
# hoststring=""
# for node in "${nodes[@]}"; do
#     hoststring+="${node}:1,"
# done
# hoststring=${hoststring%,}  # Remove the trailing comma

# echo "Running on hosts: $hoststring"

# module load mpi/openmpi-4.1.5
# module load nccl/2.18.1-cuda11.8
# # Run with Horovod
# horovodrun -np $SLURM_NNODES -H $hoststring python cdvae/run.py data=$1 expname=$2 max_num_atoms=$3