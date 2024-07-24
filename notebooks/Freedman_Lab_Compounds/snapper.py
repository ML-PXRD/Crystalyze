# %%
import yaml

# Load the YAML configuration
config_path = '/home/gridsan/tmackey/cdvae/scripts/1-22-2024_clean_impelementations/Freedman_Lab_Compounds/snapping_meta_conf.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

print("yaml loaded")

import json
import torch

# Load the configuration file
with open('snapping_parameter_dictionaries.json', 'r') as f:
    parameter_dictionaries = json.load(f)

# Convert specific values back to their required types
for param in parameter_dictionaries:
    param['UVW'] = torch.tensor(param['UVW'], device='cuda:0')

print("json loaded")

import pandas as pd
import numpy as np
import torch
import os
import sys
sys.path.append('/home/gridsan/tmackey/cdvae/scripts/1-22-2024_clean_impelementations')

import full_structure_snap as fss
import snap_utils
import importlib
importlib.reload(fss)

importlib.reload(snap_utils)
from snap_utils import * 

from pymatgen.core import Structure

import matplotlib.pyplot as plt
from pymatgen.io.cif import CifWriter

# Accessing and using values from the YAML file
model_path = config['model_path']
data_type = config['type']
data_path = config['data_path_template'].format(type=data_type)
master_results_dir = config['master_results_dir_template'].format(type=data_type)
data_name = config['data_name'].format(type=data_type)
input_patterns_file = os.path.join(data_path, config['input_patterns_file'])
input_patterns = torch.load(input_patterns_file)
num_evals = config['num_evals']
wavelength = config['wavelength']
q_max = config['q_max']
q_min = config['q_min']
num_steps = config['num_steps']
iterations = config['iterations']

for param in parameter_dictionaries:
    param['iterations'] = iterations

# %% 
for eval_num in range(num_evals):
    if eval_num == 0: 
        result_name = f"eval_recon_{data_name}.pt"
    else: 
        result_name = f"eval_recon_{data_name}_{eval_num}.pt"

    os.makedirs(master_results_dir + "_snapped/", exist_ok = True)

    snapped_compounds_dir = master_results_dir + f"_snapped/{eval_num}/"

    if iterations < 10: 
        os.makedirs(master_results_dir + "_pseudo_snapped/", exist_ok = True)
        snapped_compounds_dir = master_results_dir + f"_pseudo_snapped/{eval_num}/"

    os.makedirs(snapped_compounds_dir, exist_ok = True)
    recon = torch.load(os.path.join(model_path, result_name))

    plot_dictionary = {'plot_progress': True,
                            'plot_freq': 1000,
                            'graph_losses': False}
    
    SS = fss.FullStructureSnapper(q_max= q_max, q_min = q_min, wavelength= wavelength, plot_dictionary = plot_dictionary, parameter_dictionaries = parameter_dictionaries)

    batch_size = 2
    UVW = torch.tensor([[0,0,0.01]]).cuda()

    number_of_batches_total = np.ceil((len(recon['num_atoms'][0])) / batch_size)

    current_crystal_index = 0
    current_atom_index = 0

    for i in range(int(number_of_batches_total)):    
        
        batch = batch_processing(recon, current_crystal_index, current_atom_index, batch_size)

        pre_snapped_patterns = SS.batch_split_bin_pattern_theta(SS.batch_split_diffraction_calc(batch, 2), 2, UVW = UVW, num_steps = 8500)
        # list_of_xrds = [input_patterns[f'freedman_lab_{i}'] for i in range(current_crystal_index, min(current_crystal_index + batch_size, len(input_patterns)))]
        #list_of_ids = [f'freedman_lab_{i}' for i in range(current_crystal_index, min(current_crystal_index + batch_size, len(input_patterns)))]
        
        list_of_xrds = [input_patterns[list(input_patterns.keys())[i]] for i in range(current_crystal_index, min(current_crystal_index + batch_size, len(input_patterns)))]
        list_of_ids = [list(input_patterns.keys())[i] for i in range(current_crystal_index, min(current_crystal_index + batch_size, len(input_patterns)))]

        xrd_inputs = torch.cat(list_of_xrds)

        print(xrd_inputs)
        print(list_of_ids)
        print(batch)
        try: 
            snapped_batch, losses = SS.full_structure_snap(xrd_inputs, batch)
        except:
            print("Error in snapping")
            snapped_batch = batch
            losses = torch.zeros(batch_size) + 1

        post_snapped_patterns = SS.batch_split_bin_pattern_theta(SS.batch_split_diffraction_calc(snapped_batch, 2), 2, UVW = UVW, num_steps = 8500)

        simulated_domain = np.arange(5, 90, 0.01)

        current_sub_batch_atom_index = 0
        
        for sub_batch_index in range(batch_size): 
            
            xrd_input = xrd_inputs[sub_batch_index] / torch.max(xrd_inputs[sub_batch_index])
            pre_snapped_pattern = pre_snapped_patterns[sub_batch_index] / torch.max(pre_snapped_patterns[sub_batch_index])
            post_snapped_pattern = post_snapped_patterns[sub_batch_index] / torch.max(post_snapped_patterns[sub_batch_index])
        
            final_loss = losses[sub_batch_index]
        
            post_snapped_crystal = batch_processing(snapped_batch, sub_batch_index, current_sub_batch_atom_index, 1)
            pre_snapped_crystal = batch_processing(batch, sub_batch_index, current_sub_batch_atom_index, 1)
        
            current_sub_batch_atom_index += torch.sum(pre_snapped_crystal['num_atoms'])
                
            filename = list_of_ids[sub_batch_index]
            plot_and_save(simulated_domain, xrd_input, pre_snapped_pattern, post_snapped_pattern, snapped_compounds_dir, filename)
            save_data(xrd_input, pre_snapped_pattern, post_snapped_pattern, snapped_compounds_dir, filename, final_loss, pre_snapped_crystal, post_snapped_crystal, SS)
                
        current_crystal_index += batch_size
        current_atom_index += torch.sum(batch['num_atoms'])