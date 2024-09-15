import torch
import torch.nn as nn
import torch.nn.functional as F

from cdvae.pl_modules.embeddings import MAX_ATOMIC_NUM
from cdvae.pl_modules.gemnet.gemnet import GemNetT
from cdvae.pl_modules.model import CDVAE

def build_mlp(in_dim, hidden_dim, fc_num_layers, out_dim):
    mods = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for i in range(fc_num_layers-1):
        mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    mods += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*mods)


class GemNetTDecoder(nn.Module):
    """Decoder with GemNetT."""

    def __init__(
        self,
        hidden_dim=128,
        latent_dim=256,
        max_neighbors=20,
        radius=6.,
        scale_file=None,
    ):
        super(GemNetTDecoder, self).__init__()
        self.cutoff = radius
        self.max_num_neighbors = max_neighbors

        self.gemnet = GemNetT(
            num_targets=1,
            latent_dim=latent_dim,
            emb_size_atom=hidden_dim,
            emb_size_edge=hidden_dim,
            regress_forces=True,
            cutoff=self.cutoff,
            max_neighbors=self.max_num_neighbors,
            otf_graph=True,
            scale_file=scale_file,
        )
        self.fc_atom = nn.Linear(hidden_dim, MAX_ATOMIC_NUM)

    def forward(self, z, pred_frac_coords, pred_atom_types, num_atoms,
                lengths, angles, gt_elements = None, dropout = 0, is_training = False):
        """
        args:
            z: (N_cryst, num_latent)
            pred_frac_coords: (N_atoms, 3)
            pred_atom_types: (N_atoms, ), need to use atomic number e.g. H = 1
            num_atoms: (N_cryst,)
            lengths: (N_cryst, 3)
            angles: (N_cryst, 3)
        returns:
            atom_frac_coords: (N_atoms, 3)
            atom_types: (N_atoms, MAX_ATOMIC_NUM)
        """
        # (num_atoms, hidden_dim) (num_crysts, 3)
        h, pred_cart_coord_diff = self.gemnet(
            z=z,
            frac_coords=pred_frac_coords,
            atom_types=pred_atom_types,
            num_atoms=num_atoms,
            lengths=lengths,
            angles=angles,
            edge_index=None,
            to_jimages=None,
            num_bonds=None,
            dropout_rate = dropout, 
            training = is_training
        )
        pred_atom_types = self.fc_atom(h)
        if gt_elements is not None: 
            #impose the composition constraint
            pred_atom_types = composition_constraint(gt_elements, num_atoms, pred_atom_types)

        return pred_cart_coord_diff, pred_atom_types
    
def composition_constraint(atom_types, num_atoms, composition_per_atom):
        """
        Restrict the probability distribution from which the atom types are randomly drawn 
        to only include the elements that are present in the crystal.

        atom_types: the atom types in the crystal
        num_atoms: the number of atoms in the crystal
        composition_per_atom: the probability distribution from which the atom types are randomly drawn 

        """

        # Create a range tensor and repeat it as before
        range_tensor = torch.arange(len(num_atoms), device=num_atoms.device)
        crystal_ids = torch.repeat_interleave(range_tensor, num_atoms)

        # Convert atom_types into a mask
        atom_mask = atom_types != 0

        # For each unique crystal_id, get its corresponding indices in composition_per_atom
        unique_crystal_ids, counts = torch.unique(crystal_ids, return_counts=True)
        composition_per_atom = composition_per_atom + 1

        start_idx = 0
        for u_id, count in zip(unique_crystal_ids, counts):
            relevant_elements = atom_types[u_id][atom_mask[u_id]]

            #first step: create a hugely negative additive mask 
            mask = torch.ones_like(composition_per_atom[start_idx])
            mask *= (-10**6) # creating a matrix like [-10^6, ..., -10^6]
            mask[relevant_elements-1] = 0 # setting the elements that are present in the crystal to 0

            # second step: create a second additive mask that is used to boost any small scores for the correct elements
            additive_mask_for_normalization = torch.zeros_like(composition_per_atom[start_idx]) # creating a matrix like [0, ..., 0]
            additive_mask_for_normalization[relevant_elements-1] = 0.0001 # setting the elements that are present in the crystal to 0.0001

            # Apply masks to the relevant segment of composition_per_atom
            composition_per_atom[start_idx:start_idx+count] += mask # adding the huge negative mask to the relevant segment of composition_per_atom
            composition_per_atom[start_idx:start_idx+count] += additive_mask_for_normalization # adding the small additive mask to the relevant segment of composition_per_atom

            # Update start index for next iteration
            start_idx += count
        #print the final result after the mask constraint 
        return composition_per_atom