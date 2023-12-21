#!/bin/bash

# Capture the second argument
export JOB_NAME="$2"

# Run the main script with sbatch, setting the job name, output, and error files using the exported variable
#note: the third argument is the number of batches to run
sbatch --job-name="${JOB_NAME}" --output="output/${JOB_NAME}_%j_output.txt" --error="error/${JOB_NAME}_%j_error.txt" --export=ALL my_python_runner_gpu_eval.sh "$1" "$3" "$4" "$5"
