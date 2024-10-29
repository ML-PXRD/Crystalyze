import numpy as np
from cdvae.common.solution_utils import solve_pxrd
from cdvae.common.full_structure_snap import FullStructureSnapper
import sys
import torch
import pandas as pd

data_path = sys.argv[1]
solution_name = 'demo'
num_attempts = 2
elements = sys.argv[2:]
batch_list = []

#solve_pxrd(data_path, elements, '/home/gridsan/eriesel/hydra/singlerun/2024-10-09/tutorial_train', number_of_attempts = num_attempts, solution_name= solution_name)

for i in range(num_attempts):
    if i == 0:
        batch = torch.load('/home/gridsan/eriesel/hydra/singlerun/2024-10-09/tutorial_train/eval_recon_' + solution_name + '.pt')
    else:
        batch = torch.load('/home/gridsan/eriesel/hydra/singlerun/2024-10-09/tutorial_train/eval_recon_' + solution_name + '_' + str(i) + '.pt')
    batch['lengths'] = batch['lengths'][0]
    batch['angles'] = batch['angles'][0]
    batch['num_atoms'] = batch['num_atoms'][0]
    batch['frac_coords'] = batch['frac_coords'][0]
    batch['atom_types'] = batch['atom_types'][0]

    batch_list.append(batch)

q_max = 5.76722732255
q_min = 0.355794747244
parameter_dictionaries = [
                {'batch_size': 16,
                'bin_batch_size': 8,
                'learning_rate': 0.0004,
                'iterations': 30001,
                'UVW': torch.tensor([0,0,1.5], device='cuda:0'),
                'snap_lengths': True,
                'snap_angles': False,
                'snap_coords': False,
                'snap_UVW': [False, False, True],
                'symmeterize': True,
                'force_angles': False,
                'slow_peak_shape': False,
                'convergence_tolerance': -0.001,
                'UVW_exponent': 1.5,
                'UVW_rate_slow': 1,
                'UVW_rate': 5,
                'finished': False
                },
                {'batch_size': 16,
                'bin_batch_size': 8,
                'learning_rate': 0.0004,
                'iterations': 30001,
                'UVW': torch.tensor([0,0,1.5], device='cuda:0'),
                'snap_lengths': True,
                'snap_angles': False,
                'snap_coords': True,
                'snap_UVW': [False, False, True],
                'symmeterize': True,
                'force_angles': False,
                'slow_peak_shape': False,
                'convergence_tolerance': -0.001,
                'UVW_exponent': 1.5,
                'UVW_rate_slow': 1,
                'UVW_rate': 5,
                'finished': True
                }
                ]




fss = FullStructureSnapper(q_min = q_min, q_max = q_max, wavelength = 1.5406, in_dim = 8500, output_dim = 256, scn = False, parameter_dictionaries = parameter_dictionaries, plot_dictionary = None)
gt_pattern = torch.load('/home/gridsan/groups/Freedman_CDVAE/Crystalyze/data/demo/test_pv_xrd.pt')
gt_pattern = gt_pattern['key'].unsqueeze(0).repeat(num_attempts, 1)
total_batch = fss.merge_batch(batch_list)
snapped_batch, final_loss = fss.full_structure_snap(gt_pattern, total_batch)


batch_list = fss.split_batch(snapped_batch, 1)
for i in range(len(batch_list)):
    filename = '/home/gridsan/eriesel/hydra/singlerun/2024-10-09/tutorial_train/snapped_structures/' + str(i) + '.cif'
    structure = fss.batch_to_structure(batch_list[i])
    structure.to(filename=filename, fmt='cif')
