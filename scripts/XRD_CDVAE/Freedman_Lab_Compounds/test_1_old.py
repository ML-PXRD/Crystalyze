#%% 
import full_structure_snap as fss
import numpy as np
from math import pi
import os
from pymatgen.core.structure import Structure
import torch
import matplotlib.pyplot as plt
from pymatgen.io.cif import CifWriter
import sys

# %% 
try: 
    label = sys.argv[1]
    path_for_data = sys.argv[2]
    number_of_batches = int(sys.argv[3])
except: 
    label, path_for_data, number_of_batches = ["RRUFF", "/home/gridsan/tmackey/cdvae/data/RRUFF_data_test_only_using_amcsd/", int("64")]

print("I got past the importa")

# %% 

#for i in range(number_of_batches):
i = 0

wavelength = 1.54184
q_max = 5.76722732255
q_min = 0.355794747244
num_steps = 8500

home_directory = "/home/gridsan/tmackey/cdvae/scripts/1-22-2024_clean_impelementations/working_StructSnap_beta"
results_directory = f"{home_directory}/test_results_{i}_{label}/"

if not os.path.exists(results_directory):
    os.makedirs(results_directory)

if label != "":
    structure_directory = f"{home_directory}/test_{i}_structures_{label}/"
else:
    structure_directory = f"{home_directory}/test_{i}_structures/"

start_2theta = np.arcsin((q_min * wavelength) / (4 * pi)) * 360 / pi
stop_2theta = np.arcsin((q_max * wavelength) / (4 * pi)) * 360 / pi
step_size = (stop_2theta - start_2theta) / num_steps

simulated_domain = np.arange(num_steps) * step_size + start_2theta

np.savetxt(results_directory + "simulated_domain.txt", simulated_domain)

print("The pattern will be accurate out to:", np.arcsin((q_max * wavelength) / (4 * pi)) * 360 / pi)

plot_dictionary = {'plot_progress': True,
                    'plot_freq': 1000,
                    'graph_losses': True}

SS = fss.FullStructureSnapper(q_max= q_max, q_min = q_min, wavelength= wavelength, plot_dictionary = plot_dictionary)

skip_list = []
for filename in os.listdir(results_directory):
    if filename.split(".")[-1] == "png":
        skip_list.append(filename.split(".")[0])

losses_file = open(results_directory + "losses.txt", "a")

from pathlib import Path
path_for_xrds = os.path.join(path_for_data + "test_pv_xrd.pt")
path_for_csv = os.path.join(path_for_data + "test.csv")
data_path = Path(path_for_xrds)
data = torch.load(data_path)

import pandas as pd

test_csv = pd.read_csv(path_for_csv)
if label == "5b":
    test_batches = [1, 7, 10, 11, 17]
else:
    test_batches = [0, 2, 3, 4, 5, 6, 8, 9, 12, 13, 14, 15, 16, 18, 19]

batch_size = 256
#split the test data into batches
test_csv['batch'] = test_csv.index // batch_size 
test_csv_only_batches = test_csv[test_csv['batch'].isin(test_batches)]

gt_cif_list = []
for cif in os.listdir(structure_directory):
    if cif.split("_")[0] == "gt":
        gt_cif_list.append(cif)

UVW = torch.tensor([[0,0,0.01]]).cuda()
cif_batch_size = 2

#find the number of cif_batches that are in the gt_cif_list
num_cif_batches = len(gt_cif_list) // cif_batch_size

for i in range(num_cif_batches):
    cif_batch = gt_cif_list[i * cif_batch_size: (i + 1) * cif_batch_size]

    gt_batches = []
    for cif in cif_batch:
        gt_structure = Structure.from_file(structure_directory + f"{cif}")
        gt_batch = SS.structure_to_batch(gt_structure)
        gt_batches.append(gt_batch)
    
    batch = SS.merge_batch(gt_batches)
            
    gt_patterns = SS.batch_split_bin_pattern_theta(SS.batch_split_diffraction_calc(batch, cif_batch_size), bin_cif_batch_size, UVW = UVW, num_steps = num_steps)

    #this is for experimental PDFs, will add back in later 
    # index = int(cif.split("_")[2][:-4])
    # material_id = test_csv_only_batches['material_id'].iloc[index]
    # if label == "5b":
    #     material_id += "_0"
    # expt_pv_xrd = data[material_id].cuda()

    # #look for files that also have the index suffix 
    # for file in os.listdir(structure_directory): 
    #     if file.split("_")[0] == "pred" and file.split("_")[2][:-4] == cif.split("_")[2][:-4]:
    #         pred_structure = Structure.from_file(structure_directory + file)
    #gt_pattern = expt_pv_xrd

    pred_batches = []
    for cif in cif_batch:
        if len(cif.split("'")) > 1:
            cif = cif.split("'")[1]
            cif = cif[2:]
            cif = "'pred" + cif + "'"
        else:
            cif = cif[2:]
            cif = "pred" + cif
        #look for files that also have the index suffix 
        for file in os.listdir(structure_directory): 
            if file.split("_")[0] == "pred" and file.split("_")[2][:-4] == cif.split("_")[2][:-4]:
                pred_structure = Structure.from_file(structure_directory + file)
        pred_batch = SS.structure_to_batch(pred_structure)
        pred_batches.append(pred_batch)
    batch = SS.merge_batch(pred_batches)
# %% 
    snapped_batch, final_losses = SS.full_structure_snap(gt_patterns, batch)
    print(final_losses)
# %%
