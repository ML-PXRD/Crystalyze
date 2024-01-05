#!/bin/bash

# Initialize default values for the variables
model_path=""
job_name=""
num_batches=""
force_types=""
num_evals=""
test_set_override=""

# Process each argument
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_path) model_path="$2"; shift ;;
        --job_name) job_name="$2"; export JOB_NAME="$2"; shift ;;
        --num_batches) num_batches="$2"; shift ;;
        --force_types) force_types="$2"; shift ;;
        --num_evals) num_evals="$2"; shift ;;
        --test_set_override) test_set_override="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Check if mandatory parameters are set
if [ -z "$job_name" ] || [ -z "$model_path" ]; then
    echo "Error: job_name and model_path are required."
    exit 1
fi

# Construct the sbatch command
sbatch_cmd="sbatch --job-name=\"${JOB_NAME}\" \
             --output=\"output/${JOB_NAME}_%j_output.txt\" \
             --error=\"error/${JOB_NAME}_%j_error.txt\" \
             --export=ALL \
             my_python_runner_gpu_eval.sh \"$model_path\" \"$num_batches\" \"$force_types\" \"$num_evals\" \"$test_set_override\"" 

# Display the sbatch command
echo "Executing: $sbatch_cmd"

# Run the sbatch command
eval "$sbatch_cmd"

#example usage:
#./eval_launcher_with_flags.sh --model_path "/path/to/model" --job_name "example_job" --num_batches 5 --force_types "type1" --num_evals 10
