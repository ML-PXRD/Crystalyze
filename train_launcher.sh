#!/bin/bash

# Capture the third argument as the job name and fourth as max_num_atoms
export JOB_NAME="$3"
export MAX_NUM_ATOMS="$4"

# Run the main script with sbatch, also passing max_num_atoms
sbatch --job-name="${JOB_NAME}" --output="output/${JOB_NAME}_%N_%j_output.txt" --error="error/${JOB_NAME}_%N_%j_error.txt" --export=ALL my_python_runner_gpu_train.sh "$1" "$2" "$MAX_NUM_ATOMS"
