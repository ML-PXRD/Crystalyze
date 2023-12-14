 
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

import pandas as pd

from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.diffraction.xrd import XRDCalculator

import tqdm 

all_frac_coords_stack = []
all_atom_types_stack = []
frac_coords = []
num_atoms = []
atom_types = []
lengths = []
angles = []
input_data_list = []

#verifying on perov_5

list_of_idxs = []
list_of_batchs = []
model_path = Path("/home/gridsan/tmackey/hydra/singlerun/2023-10-03/CondPerovV13")
model, test_loader, cfg = load_model(model_path, True)

loader = test_loader

def generate_unique_id(species, frac_coords, a,b,c, alpha, beta, gamma):
    atomic_species = [str(specie) for specie in species]
    frac_coords = [coord for sublist in frac_coords for coord in sublist]
    frac_coords = [round(coord, 7) for coord in frac_coords]
    frac_coords_str = ''.join(map(str, frac_coords))
    lattice_lengths = [a,b,c]
    lattice_angles = [alpha, beta, gamma]
    lattice_lengths = [round(length, 7) for length in lattice_lengths]
    lattice_angles = [round(angle, 7) for angle in lattice_angles]

    lattice_str = ''.join(map(str, lattice_lengths + lattice_angles))
    # combined_data = ''.join(atomic_species) + frac_coords_str + lattice_str
    combined_data = ''.join(atomic_species) + lattice_str
        
    return combined_data

for idx, batch in enumerate(loader):
    print(idx)

    list_verion = batch.to_data_list()

    #for data in list_verion:
    #use tqdm 

    for data in tqdm.tqdm(list_verion):
        peak_locations = []
        species = [Element.from_Z(atom_type.item()).symbol for atom_type in data.atom_types.cpu()]
        frac_coordinates = data.frac_coords.cpu().tolist()
        angles = data.angles.cpu().tolist()
        lengths = data.lengths.cpu().tolist()
        alpha, beta, gamma = angles[0]
        a, b, c = lengths[0]

        test_unique_ID = generate_unique_id(species, frac_coordinates, a,b,c, alpha, beta, gamma)

        folder_dir = '/home/gridsan/tmackey/cdvae/data/perov_5/'

        df_train = pd.read_csv(folder_dir + 'train_xrd.csv')
        df_val = pd.read_csv(folder_dir + 'val_xrd.csv')
        df_test = pd.read_csv(folder_dir + 'test_xrd.csv')
        df_all = pd.concat([df_train, df_val, df_test])

        #convert the dataframe to a dictionary for faster lookups. the key is the TsachID and the value is the 'xrd' entry
        df_all_dict = df_all.set_index('TsachID').to_dict()['xrd']

        try: 
            xrd_pattern = df_all_dict[test_unique_ID]
        except:
            print('There are no rows with the same TsachID')
            print('The TsachID is {}'.format(test_unique_ID))
            print('The data is {}'.format(data))
            print('The frac_coords are {}'.format(frac_coordinates))
            print('The species are {}'.format(species))
            print('The angles are {}'.format(angles))
            print('The lengths are {}'.format(lengths))

