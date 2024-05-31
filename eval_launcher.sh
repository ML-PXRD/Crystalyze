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
        --label) label="$2"; shift ;;
        --force_num_atoms) force_num_atoms="$2"; shift ;;
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
             my_python_runner_gpu_eval.sh \"$model_path\" \"$num_batches\" \"$force_types\" \"$num_evals\" \"$test_set_override\" \"$label\" \"$force_num_atoms\" "

# Display the sbatch command
echo "Executing: $sbatch_cmd"

# Run the sbatch command
eval "$sbatch_cmd"

#example usage:
#./eval_launcher_with_flags.sh --model_path "/path/to/model" --job_name "example_job" --num_batches 5 --force_types "type1" --num_evals 10

#bash eval_launcher.sh --model_path "/home/gridsan/tmackey/hydra/singlerun/2024-01-29/vae_nopf/" --job_name "wstoich_vae_nopf_eval" --num_batches 100 --force_types "True" --num_evals 10 --test_set_override "" --label "wstoich_full_set_10_evals" --force_num_atoms "True"
#bash eval_launcher.sh --model_path "/home/gridsan/tmackey/hydra/singlerun/2024-01-29/vae_nopf/" --job_name "wstoich_vae_nopf_eval" --num_batches 5 --force_types "True" --num_evals 64 --test_set_override "" --label "wstoich_5batches_64_evals" --force_num_atoms "True"
#bash eval_launcher.sh --model_path "/home/gridsan/tmackey/hydra/singlerun/2024-01-31/ae_pf/" --job_name "wstoich_ae_pf" --num_batches 100 --force_types "True" --num_evals 10 --test_set_override "" --label "wstoich_full_set_10_evals" --force_num_atoms "True"
#bash eval_launcher.sh --model_path "/home/gridsan/tmackey/hydra/singlerun/2024-01-31/ae_pf/" --job_name "wstoich_ae_pf" --num_batches 5 --force_types "True" --num_evals 64 --test_set_override "" --label "wstoich_5batches_64_evals" --force_num_atoms "True"
#bash eval_launcher.sh --model_path "/home/gridsan/tmackey/hydra/singlerun/2024-01-31/ae_pf/" --job_name "unsolved_wstoich_ae_pf" --num_batches 1 --force_types "True" --num_evals 64 --test_set_override "unsolved_compounds" --label "wstoich_unsolved_compounds" --force_num_atoms "True"
#bash eval_launcher.sh --model_path "/home/gridsan/tmackey/hydra/singlerun/2024-01-29/augmented_vae_nopf" --job_name "wstoich_RRUFF" --num_batches 1 --force_types "True" --num_evals 64 --test_set_override "RRUFF_data_test_only_using_amcsd" --label "wstoich_RRUFF" --force_num_atoms "True"
#bash eval_launcher.sh --model_path "/home/gridsan/tmackey/hydra/singlerun/2024-01-29/vae_nopf" --job_name "wstoich_RRUFF" --num_batches 1 --force_types "True" --num_evals 64 --test_set_override "RRUFF_data_test_only_using_amcsd" --label "wstoich_RRUFF" --force_num_atoms "True"
#bash eval_launcher.sh --model_path "/home/gridsan/tmackey/hydra/singlerun/2024-01-29/augmented_vae_nopf" --job_name "wstoich_PDF" --num_batches 1 --force_types "True" --num_evals 64 --test_set_override "PDF_unsolved_compounds" --label "wstoich_PDF" --force_num_atoms "True"
#sbatch eval_launcher.sh --model_path "/home/gridsan/tmackey/hydra/singlerun/2024-01-29/augmented_vae_nopf" --job_name "Freedman_lab_best_K_Bi_files" --num_batches 1 --force_types "False" --num_evals 64 --test_set_override "Freedman_lab_best_K_Bi_files" --label "Freedman_lab_best_K_Bi_files" --force_num_atoms "False"
# sbatch eval_launcher.sh --model_path "/home/gridsan/tmackey/hydra/singlerun/2024-01-29/augmented_vae_nopf" --job_name "Freedman_lab_round_3_Tr" --num_batches 1 --force_types "True" --num_evals 64 --test_set_override "Freedman_lab_round_3_Tr" --label "Freedman_lab_round_3_Tr" --force_num_atoms "True"
# sbatch eval_launcher.sh --model_path "/home/gridsan/tmackey/hydra/singlerun/2024-01-29/augmented_vae_nopf" --job_name "Freedman_lab_round_3_Fa" --num_batches 1 --force_types "False" --num_evals 64 --test_set_override "Freedman_lab_round_3_Fa" --label "Freedman_lab_round_3_Fa" --force_num_atoms "False"
# sbatch eval_launcher.sh --model_path "/home/gridsan/tmackey/hydra/singlerun/2024-01-29/augmented_vae_nopf" --job_name "PDF_filtered_raw_newly_subtracted" --num_batches 64 --force_types "False" --num_evals 64 --test_set_override "PDF_filtered_raw_newly_subtracted" --label "PDF_filtered_raw_newly_subtracted" --force_num_atoms "False"
# sbatch eval_launcher.sh --model_path "/home/gridsan/tmackey/hydra/singlerun/2024-01-29/augmented_vae_nopf" --job_name "PDF_filtered_raw_newly_subtracted" --num_batches 64 --force_types "False" --num_evals 64 --test_set_override "PDF_filtered_raw_newly_subtracted" --label "PDF_filtered_raw_newly_subtracted" --force_num_atoms "False"
# sbatch eval_launcher.sh --model_path "/home/gridsan/tmackey/hydra/singlerun/2024-01-29/augmented_vae_nopf" --job_name "Freedman_lab_diffraction_for_Tsach_newly_subtracted" --num_batches 64 --force_types "False" --num_evals 64 --test_set_override "Freedman_lab_diffraction_for_Tsach_newly_subtracted" --label "Freedman_lab_diffraction_for_Tsach_newly_subtracted" --force_num_atoms "False"
# sbatch eval_launcher.sh --model_path "/home/gridsan/tmackey/hydra/singlerun/2024-01-29/augmented_vae_nopf" --job_name "CoBi" --num_batches 64 --force_types "False" --num_evals 64 --test_set_override "CoBi" --label "CoBi" --force_num_atoms "False"
