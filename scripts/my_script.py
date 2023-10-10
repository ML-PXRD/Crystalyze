 
import time
import argparse
import torch

from tqdm import tqdm
from torch.optim import Adam
from pathlib import Path
from types import SimpleNamespace
from torch_geometric.data import Batch
from torch_geometric.data.separate import separate

#import a library that allows you to reload a module
from importlib import reload

from eval_utils import load_model

all_frac_coords_stack = []
all_atom_types_stack = []
frac_coords = []
num_atoms = []
atom_types = []
lengths = []
angles = []
input_data_list = []

#my code 
list_of_idxs = []
list_of_batchs = []
model_path = Path("/home/gridsan/tmackey/hydra/singlerun/2023-09-13/perov")
model, test_loader, cfg = load_model(model_path, True)

loader = test_loader

for idx, batch in enumerate(loader):
    print(idx)
    print(batch)
    list_of_idxs.append(idx)
    list_of_batchs.append(batch)

idx = list_of_idxs[0]
batch = list_of_batchs[0]

batch.cuda()
print(f'batch {idx} in {len(loader)}')
batch_all_frac_coords = []
batch_all_atom_types = []
batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
batch_lengths, batch_angles = [], []

# only sample one z, multiple evals for stoichaticity in langevin dynamics
model.to(0)

_, _, z = model.encode(batch)

num_evals = 1
eval_idx = 0

gt_num_atoms = batch.num_atoms 
gt_atom_types = batch.atom_types

ld_kwargs = SimpleNamespace(n_step_each=100,step_lr=1e-4,min_sigma=0,save_traj=False,disable_bar=True)

# outputs = model.langevin_dynamics(z, ld_kwargs, gt_num_atoms, gt_atom_types)

# # collect sampled crystals in this batch
# batch_frac_coords.append(outputs['frac_coords'].detach().cpu())
# batch_num_atoms.append(outputs['num_atoms'].detach().cpu())
# batch_atom_types.append(outputs['atom_types'].detach().cpu())
# batch_lengths.append(outputs['lengths'].detach().cpu())
# batch_angles.append(outputs['angles'].detach().cpu())

# # collect sampled crystals for this z.
# frac_coords.append(torch.stack(batch_frac_coords, dim=0))
# num_atoms.append(torch.stack(batch_num_atoms, dim=0))
# atom_types.append(torch.stack(batch_atom_types, dim=0))
# lengths.append(torch.stack(batch_lengths, dim=0))
# angles.append(torch.stack(batch_angles, dim=0))


#doing the pxrd stuff 
from pymatgen.core.structure import Structure
atomic_species = [atom_type for atom_type in list_verion[1].atom_types]
data = list_verion[1]
frac_coordinates = data.frac_coords.tolist()

print(batch)

empty_attributes = []
empty_attributes = []
alpha, beta, gamma = data.angles[0]
a, b, c = data.lengths[0]

from pymatgen.core.lattice import Lattice
lattice = Lattice.from_parameters(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
structure = Structure(lattice, species=atomic_species, coords=frac_coordinates, coords_are_cartesian=False)
from pymatgen.analysis.diffraction.xrd import XRDCalculator
xrd_calculator = XRDCalculator(wavelength='CuKa', symprec=0.1)
pattern = xrd_calculator.get_pattern(structure)

for attr in dir(batch):
    value = getattr(batch, attr)
    if not callable(value) and not attr.startswith("__"):
        if isinstance(value, list) and not value:
            empty_attributes.append(attr)
        elif isinstance(value, dict) and not value:
            empty_attributes.append(attr)
        elif isinstance(value, str) and not value:
            empty_attributes.append(attr)
        elif isinstance(value, torch.Tensor) and value.nelement() == 0:  # Checks if the tensor is empty
            empty_attributes.append(attr)
        elif value is None:
            empty_attributes.append(attr)

print("Empty attributes:", empty_attributes)

attributes_to_remove = empty_attributes

for attr in attributes_to_remove:
    if hasattr(batch, attr):
        delattr(batch, attr)

for attr in empty_attributes:
    if hasattr(batch, attr):
        setattr(batch, attr, 1)


# Save the ground truth structure
input_data_list = input_data_list + batch.to_data_list()

[batch.get(i) for i in range(batch.num_graphs)]
from torch_geometric.data.separate import separate


#reload the separate module
reload(torch_geometric.data.separate)

data = separate(
            cls=batch.__class__.__bases__[-1],
            batch=batch,
            idx=idx,
            slice_dict=batch._slice_dict,
            inc_dict=batch._inc_dict,
            decrement=False,
        )

for attr in attrs: 
    if inc_dict[attr] is None:
        print(attr)
        print(inc_dict[attr])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--tasks', nargs='+', default=['recon', 'gen', 'opt'])
    parser.add_argument('--n_step_each', default=100, type=int)
    parser.add_argument('--step_lr', default=1e-4, type=float)
    parser.add_argument('--min_sigma', default=0, type=float)
    parser.add_argument('--save_traj', default=False, type=bool)
    parser.add_argument('--disable_bar', default=False, type=bool)
    parser.add_argument('--num_evals', default=1, type=int)
    parser.add_argument('--num_batches_to_samples', default=20, type=int)
    parser.add_argument('--start_from', default='data', type=str)
    parser.add_argument('--batch_size', default=500, type=int)
    parser.add_argument('--force_num_atoms', action='store_true')
    parser.add_argument('--force_atom_types', action='store_true')
    parser.add_argument('--down_sample_traj_step', default=10, type=int)
    parser.add_argument('--label', default='')

    args = parser.parse_args()

    main(args)
