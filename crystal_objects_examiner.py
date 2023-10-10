# %%
from collections import Counter
import argparse
import os
import json

import numpy as np
from pathlib import Path
from tqdm import tqdm
from p_tqdm import p_map
from scipy.stats import wasserstein_distance

from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.structure_matcher import StructureMatcher
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from matminer.featurizers.composition.composite import ElementProperty
import compute_metrics

import time
import argparse
import torch

from tqdm import tqdm
from torch.optim import Adam
from pathlib import Path
from types import SimpleNamespace
from torch_geometric.data import Batch

from eval_utils import load_model
import evaluate
# %% 

model_path = Path("/home/gridsan/tmackey/hydra/singlerun/2023-09-13/perov")
    
model, test_loader, cfg = load_model(model_path, load_data=True)

print(test_loader)

ld_kwargs = SimpleNamespace(n_step_each=100,step_lr=1e-4,min_sigma=0,save_traj=False,disable_bar=True)

if torch.cuda.is_available(): model.to('cuda')

print('Evaluate model on the reconstruction task.')
start_time = time.time()
(frac_coords, num_atoms, atom_types, lengths, angles,all_frac_coords_stack, all_atom_types_stack, input_data_batch) = evaluate.reconstructon(test_loader, model, ld_kwargs, 1,'store_true', 'store_true', 10)

if args.label == '':
    recon_out_name = 'eval_recon.pt'
else:
    recon_out_name = f'eval_recon_{args.label}.pt'

torch.save({
    'eval_setting': args,
    'input_data_batch': input_data_batch,
    'frac_coords': frac_coords,
    'num_atoms': num_atoms,
    'atom_types': atom_types,
    'lengths': lengths,
    'angles': angles,
    'all_frac_coords_stack': all_frac_coords_stack,
    'all_atom_types_stack': all_atom_types_stack,
    'time': time.time() - start_time
}, model_path / recon_out_name)


recon_file_path = "/home/gridsan/tmackey/hydra/singlerun/2023-09-13/perov/eval_recon.pt"

print(recon_file_path)

crys_array_list, true_crystal_array_list = compute_metrics.get_crystal_array_list(recon_file_path)
# the problem is with the above line of code: 
# the number of elements in the crys array list is 3000+ 
# but the number of elements in the true crystal array list is only 201 

#need to figure out why the elements in the true crystal array list is so low 
#the true crystal array list is also stored inside the pt file with the reconstructed crystal objects 
# they are stored as numpy arrays in the manner detailed in the paper 

