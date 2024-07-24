import sys
import os
import numpy as np
import math

import os
import torch
import numpy as np
from tqdm import tqdm

from pymatgen.analysis.diffraction.xrd import XRDCalculator
xrd_calculator = XRDCalculator(wavelength='CuKa')
import evaluate_utils.eval_utils as eval_utils

try: 
    worker_num = int(sys.argv[1])
    num_splits = int(sys.argv[2])
    model_path = str(sys.argv[3])
    label = str(sys.argv[4])
    num_batches = int(sys.argv[5])
    snapped_dir = str(sys.argv[6])
except: 
    worker_num = 0
    num_splits = 100
    model_path = '/home/gridsan/tmackey/hydra/singlerun/2024-01-29/vae_nopf'
    label = '5_batches_64_evals'
    num_batches = 5
    snapped_dir = '/home/gridsan/tmackey/cdvae/scripts/1-22-2024_clean_impelementations/MP_20_Testing/Snapped_Compounds'

recon_file_path = eval_utils.get_file_paths(model_path, 'recon',label=label)
all_results, all_gt = eval_utils.all_results_retreival(recon_file_path, num_batches, label = label)
split_size = math.ceil(len(all_results[0]) / num_splits)

all_results = [all_results[i][worker_num*split_size:(worker_num+1)*split_size] for i in range(num_batches)]
all_gt = [all_gt[i][worker_num*split_size:(worker_num+1)*split_size] for i in range(num_batches)]
set_size = len(all_results[0])

snapped_crystal_list = torch.load(os.path.join(snapped_dir, "snapped_crystal_list.pt"))
gt_crystal_list = torch.load(os.path.join(snapped_dir, "gt_crystal_list.pt"))

snapped_crystal_list = [snapped_crystal_list[i][worker_num*split_size:(worker_num+1)*split_size] for i in range(1)]
gt_crystal_list = [gt_crystal_list[i][worker_num*split_size:(worker_num+1)*split_size] for i in range(1)]

fs_total_rmsd, fs_total_rmsd_just_sites = eval_utils.evaluation(all_results, all_gt, set_size, num_batches = num_batches) 

ltol_values = [0.01, 0.05, 0.1, 0.2, 0.3]
angle_tol_values = [0.5, 1.0, 5.0, 10.0]

evals_per_ltol, evals_per_angle_tol = eval_utils.tolerance_analysis(all_results, all_gt, ltol_values, angle_tol_values, set_size, num_batches)

snapped_evals_per_ltol, snapped_evals_per_angle_tol = eval_utils.tolerance_analysis(snapped_crystal_list, gt_crystal_list, ltol_values, angle_tol_values, set_size, 1)

pred_spacegroups_and_tolerance = [(eval_utils.symmetry_performance(all_results[i], all_gt[i])) for i in range(num_batches)]
fs_pred_spacegroups = np.stack([x[0] for x in pred_spacegroups_and_tolerance])
fs_pred_tolerance = np.stack([x[1] for x in pred_spacegroups_and_tolerance])
fs_gt_spacegroups = np.stack([x[2] for x in pred_spacegroups_and_tolerance])
fs_total_rmsd_just_sites = np.stack(fs_total_rmsd_just_sites)

#make a custom filename for the results that looks like worker<worker_num> 
filename = "worker" + "_" + str(worker_num)

#make a directory inside of the model path folder to store the results
new_dir = model_path + "/" + label 
if not os.path.exists(new_dir):
    os.makedirs(new_dir)

# Define a dictionary to hold the variable references and their corresponding suffixes for filenames
file_suffixes = {
    "fs_total_rmsd": "_fs_total_rmsd.npy",
    "fs_pred_spacegroups": "_fs_pred_spacegroups.npy",
    "fs_gt_spacegroups": "_fs_gt_spacegroups.npy",
    "fs_pred_tolerance": "_fs_pred_tolerance.npy",
    "fs_total_rmsd_just_sites": "_fs_total_rmsd_just_sites.npy",
    "evals_per_ltol": "_evals_per_ltol.npy",
    "evals_per_angle_tol": "_evals_per_angle_tol.npy",
    "snapped_evals_per_angle_tol": "_snapped_evals_per_angle_tol.npy",
    "snapped_evals_per_ltol": "_snapped_evals_per_ltol.npy",
}

# Loop through the dictionary, convert arrays to float and save them
for var_name, suffix in file_suffixes.items():
    # Dynamically access the variable using globals() and convert to float
    globals()[var_name] = globals()[var_name].astype(float)
    
    # Construct the filename and save the array
    new_filename = f"{new_dir}/{filename}{suffix}"
    np.save(new_filename, globals()[var_name])