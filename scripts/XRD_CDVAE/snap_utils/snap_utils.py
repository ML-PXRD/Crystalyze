import numpy as np 
import matplotlib.pyplot as plt
import os
from pymatgen.io.cif import CifWriter
import full_structure_snap as fss
import torch

def plot_and_save(simulated_domain, xrd_input, pre_snapped_pattern, post_snapped_pattern, results_directory, filename):
    """ 
    Args: 

    Simulated domain is a tensor of the domain consistent across the pre-snapped pattern, post-snapped pattern and inputs 
    xrd_input is a tensor with the intensities of one of the patterns input. Should have dimensions 1x8500.
    pre_snapped_pattern is a tensor with the intensities of pre-snapped compound's pattern input. Should have dimensions 1x8500
    post_snapped_pattern is a tensor with the intensities of the post-snapped compound's pattern input. Should have dimensions 1x8500
    results_directory: where you want the plots to be saved

    """

    plt.figure()
    plt.plot(simulated_domain, xrd_input.numpy(), label='Experimental Pattern')
    plt.plot(simulated_domain, pre_snapped_pattern.cpu().numpy() + 1, label='Pre-snapped Pattern')
    plt.plot(simulated_domain, post_snapped_pattern.cpu().numpy() + 2, label='Post-snapped Pattern')
    plt.legend()
    plt.savefig(os.path.join(results_directory, f"{filename}.png"), dpi=300)

def save_data(xrd_input, pre_snapped_pattern, post_snapped_pattern, results_directory, filename, final_loss, pre_snapped_batch, post_snapped_batch, SS): 
    np.savetxt(os.path.join(results_directory, f"{filename}_gt_pattern.txt"), xrd_input.cpu().numpy())
    np.savetxt(os.path.join(results_directory, f"{filename}_pre_snapped_pattern.txt"), pre_snapped_pattern.cpu().numpy())
    np.savetxt(os.path.join(results_directory, f"{filename}_post_snapped_pattern.txt"), post_snapped_pattern.cpu().numpy())

    with open(os.path.join(results_directory, "losses.txt"), "a") as losses_file:
        losses_file.write(f"{filename},{final_loss}\n")

    pre_snapped_structure = SS.batch_to_structure(pre_snapped_batch)
    post_snapped_structure = SS.batch_to_structure(post_snapped_batch)

    CifWriter(pre_snapped_structure).write_file(os.path.join(results_directory, f"pre_snapped_{filename}.cif"))
    CifWriter(post_snapped_structure).write_file(os.path.join(results_directory, f"post_snapped_{filename}.cif"))


def batch_processing(recon, current_crystal_index, current_atom_index, batch_size):
    try: 
        num_atoms = recon['num_atoms'][0][current_crystal_index:current_crystal_index + batch_size]
        frac_coords = recon['frac_coords'][0][current_atom_index : current_atom_index + torch.sum(num_atoms)]
        lengths = recon['lengths'][0][current_crystal_index:current_crystal_index + batch_size ]
        angles = recon['angles'][0][current_crystal_index:current_crystal_index + batch_size ]
        atom_types = recon['atom_types'][0][current_atom_index : current_atom_index + torch.sum(num_atoms)]

    except: 
        num_atoms = recon['num_atoms'][current_crystal_index:current_crystal_index + batch_size]
        frac_coords = recon['frac_coords'][current_atom_index : current_atom_index + torch.sum(num_atoms)]
        lengths = recon['lengths'][current_crystal_index:current_crystal_index + batch_size ]
        angles = recon['angles'][current_crystal_index:current_crystal_index + batch_size ]
        atom_types = recon['atom_types'][current_atom_index : current_atom_index + torch.sum(num_atoms)]

    batch = {'frac_coords' : frac_coords.to('cuda:0'),
        'lengths': lengths.to('cuda:0'), 
        'angles': angles.to('cuda:0'),
        'atom_types': atom_types.to('cuda:0'),
        'num_atoms': num_atoms.to('cuda:0')
    }

    return batch
