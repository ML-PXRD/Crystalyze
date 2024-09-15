# NOTE: these are function that make doing inference on a small number of compounds easier and will 
# be integrated with an inference notebook in the future.

import numpy as np
import pandas as pd

import random
from pymatgen.core import Structure, Lattice, Element
from cdvae.common.data_utils import lattice_params_to_matrix

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis import local_env
import yaml

def create_inference_dataframe(inference_data):
    """
    Create a dataframe for inference data. Note that all columns except for material_id and atomic numbers are filled with dummy data because they are either depreciated features
    or contain information that would be be accessible in the inference process.

    Args:
        inference_data (dict): A dictionary containing inference data.

    Returns:
        pandas.DataFrame: A dataframe with columns 'cif', 'filename', 'material_id', 'atomic_nums',
        'formation_energy_per_atom', 'spacegroup.number', 'xrd_peak_intensities', 'xrd_peak_locations',
        and 'disc_sim_xrd'. 
        
    """

    #initialize an empty dataframe with 'cif' and 'filename' columns
    df = pd.DataFrame(columns = ['cif', 'filename', 'material_id', 'atomic_nums'])

    #set the 'material_id' column to the keys of the inference_data dictionary and the 'atomic_numbers' column to the values of the inference_data dictionary
    df['material_id'] = [id for id in inference_data.keys()]
    df['atomic_numbers'] = [data[1] for data in inference_data.values()]

    #the following are data values that were used in R&D but are not relevant for inference (at this point in time)

    #add the 'filename' column and set it to None
    df['filename'] = None

    #add the formation_energy_per_atom and spacegroup.number columns and just set them to 0 
    df['formation_energy_per_atom'] = 0
    df['spacegroup.number'] = 0

    #add the cifs to the 'cif' column
    df['cif'] = None

    #make a 'xrd_peak_intensities' and 'xrd_peak_locations' columns where each entry is 256 * [0]
    df['xrd_peak_intensities'] = [256 * [0] for _ in range(len(df))]
    df['xrd_peak_locations'] = [256 * [0] for _ in range(len(df))]
    df['disc_sim_xrd'] = [np.array(256 * [0]) for _ in range(len(df))]

    return df

def create_inference_xrd_data(inference_data):
    """
    Create a dictionary of XRD data to save from the given inference data.

    Parameters:
    inference_data (dict): A dictionary containing inference data, where the keys are IDs and the values are data.

    Returns:
    dict: A dictionary containing XRD data to save, where the keys are IDs and the values are the first element of the data.
    """
    xrd_data_to_save = {id:data[0] for id, data in inference_data.items()}
    return xrd_data_to_save


def create_inference_graph_data(inference_data):
    """
    Create dummy graph data for inference. This graph data does not contain any information about the coordinates or lattice parameters of the graph, it just initializes a 
    random unit cell with atoms corresponding to the atom types specified. The types and number of atoms can be used in inference or not used in inference depending on 
    the flags input at inference. 

    Args:
        inference_data (dict): A dictionary containing inference data.

    Returns:
        dict: A dictionary containing graph data for each ID in the inference data.
    """

    graph_data_dict = {}

    for id, data in inference_data.items():
        _, elements_involved = data
        
        structure = generate_random_structure(elements_involved)

        pyg_graph_data = generate_pyg_graph(structure)

        graph_data_dict[id] = pyg_graph_data

    return graph_data_dict

def generate_random_structure(atomic_numbers):
    """
    Generates a random crystal structure with the given atomic numbers.

    Parameters:
    atomic_numbers (list): A list of atomic numbers representing the elements in the structure.

    Returns:
    Structure: A random crystal structure object.

    """
    # Generate random lattice parameters
    a = random.uniform(3.0, 10.0)
    b = random.uniform(3.0, 10.0)
    c = random.uniform(3.0, 10.0)
    alpha = random.uniform(70, 110)
    beta = random.uniform(70, 110)
    gamma = random.uniform(70, 110)
    
    # Create random lattice
    lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
    
    # Generate random fractional coordinates for each atom
    coordinates = []
    species = []
    for atomic_number in atomic_numbers:
        species.append(Element.from_Z(atomic_number))
        coordinates.append([random.random(), random.random(), random.random()])
    
    # Create structure
    structure = Structure(lattice, species, coordinates)
    
    return structure

def generate_pyg_graph(crystal, graph_method='crystalnn'):
    """
    Generate a PyTorch Geometric graph representation from a crystal structure.

    Args:
        crystal (pymatgen.core.structure.Structure): The crystal structure.
        graph_method (str, optional): The method to generate the graph. Defaults to 'crystalnn'.

    Returns:
        tuple: A tuple containing the following elements:
            - frac_coords (numpy.ndarray): The fractional coordinates of the atoms.
            - atom_types (numpy.ndarray): The atomic numbers of the atoms.
            - lengths (numpy.ndarray): The lengths of the lattice parameters.
            - angles (numpy.ndarray): The angles of the lattice parameters.
            - edge_indices (numpy.ndarray): The indices of the edges in the graph.
            - to_jimages (numpy.ndarray): The translation vectors for periodic images.
            - num_atoms (int): The number of atoms in the crystal structure.
    """

    CrystalNN = local_env.CrystalNN(distance_cutoffs=None, x_diff_weight=-1, porous_adjustment=False)

    if graph_method == 'crystalnn':
        crystal_graph = StructureGraph.with_local_env_strategy(
            crystal, CrystalNN)
    elif graph_method == 'none':
        pass
    else:
        raise NotImplementedError

    frac_coords = crystal.frac_coords
    atom_types = crystal.atomic_numbers
    lattice_parameters = crystal.lattice.parameters
    lengths = lattice_parameters[:3]
    angles = lattice_parameters[3:]

    assert np.allclose(crystal.lattice.matrix,
                    lattice_params_to_matrix(*lengths, *angles))

    edge_indices, to_jimages = [], []
    if graph_method != 'none':
        for i, j, to_jimage in crystal_graph.graph.edges(data='to_jimage'):
            edge_indices.append([j, i])
            to_jimages.append(to_jimage)
            edge_indices.append([i, j])
            to_jimages.append(tuple(-tj for tj in to_jimage))

    atom_types = np.array(atom_types)
    lengths, angles = np.array(lengths), np.array(angles)
    edge_indices = np.array(edge_indices)
    to_jimages = np.array(to_jimages)
    num_atoms = atom_types.shape[0]

    return frac_coords, atom_types, lengths, angles, edge_indices, to_jimages, num_atoms



def load_and_modify_config(input_string):
    """
    Load the configuration, modify the root path based on the input string, 
    and print the updated configuration.

    Args:
        input_string (str): The input string to modify the root path.
    """

    config = {
        'root_path': '${oc.env:PROJECT_ROOT}/data/CoBi',
        'prop': 'spacegroup.number',
        'num_targets': 1,
        'niggli': True,
        'primitive': False,
        'graph_method': 'crystalnn',
        'lattice_scale_method': 'scale_length',
        'preprocess_workers': 30,
        'readout': 'mean',
        'max_atoms': 20,
        'otf_graph': False,
        'eval_model_name': f'{input_string}',
        'noise_sd': 0,
        'train_max_epochs': 250,
        'early_stopping_patience': 100000,
        'teacher_forcing_max_epoch': 125,
        'datamodule': {
            '_target_': 'cdvae.pl_data.datamodule.CrystDataModule',
            'datasets': {
                'train': {
                    '_target_': 'cdvae.pl_data.dataset.CrystDataset',
                    'name': 'Formation energy train',
                    'path': '${data.root_path}/train.csv',
                    'prop': '${data.prop}',
                    'niggli': '${data.niggli}',
                    'primitive': '${data.primitive}',
                    'graph_method': '${data.graph_method}',
                    'lattice_scale_method': '${data.lattice_scale_method}',
                    'preprocess_workers': '${data.preprocess_workers}',
                    'train_fraction': 1,
                },
                'val': [{
                    '_target_': 'cdvae.pl_data.dataset.CrystDataset',
                    'name': 'Formation energy val',
                    'path': '${data.root_path}/val.csv',
                    'prop': '${data.prop}',
                    'niggli': '${data.niggli}',
                    'primitive': '${data.primitive}',
                    'graph_method': '${data.graph_method}',
                    'lattice_scale_method': '${data.lattice_scale_method}',
                    'preprocess_workers': '${data.preprocess_workers}',
                }],
                'test': [{
                    '_target_': 'cdvae.pl_data.dataset.CrystDataset',
                    'name': 'Formation energy test',
                    'path': '${data.root_path}/test.csv',
                    'prop': '${data.prop}',
                    'niggli': '${data.niggli}',
                    'primitive': '${data.primitive}',
                    'graph_method': '${data.graph_method}',
                    'lattice_scale_method': '${data.lattice_scale_method}',
                    'preprocess_workers': '${data.preprocess_workers}',
                }],
            },
            'num_workers': {
                'train': 0,
                'val': 0,
                'test': 0,
            },
            'batch_size': {
                'train': 256,
                'val': 256,
                'test': 256,
            }
        }
    }

    # Modify root_path with the input string
    config['root_path'] = f'${{oc.env:PROJECT_ROOT}}/data/{input_string}'

    # Update paths in datasets
    datasets = config['datamodule']['datasets']
    for dataset_key in datasets:
        dataset = datasets[dataset_key]
        if isinstance(dataset, list):
            for sub_dataset in dataset:
                sub_dataset['path'] = sub_dataset['path'].replace('${data.root_path}', config['root_path'])
        else:
            dataset['path'] = dataset['path'].replace('${data.root_path}', config['root_path'])

    # Convert the config dictionary to YAML string
    config_yaml = yaml.dump(config, default_flow_style=False)

    return config_yaml