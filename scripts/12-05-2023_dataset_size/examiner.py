import numpy as np
import pandas as pd
import networkx as nx
import torch
import copy
import itertools

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis import local_env

from networkx.algorithms.components import is_connected

from sklearn.metrics import accuracy_score, recall_score, precision_score

from torch_scatter import scatter

from p_tqdm import p_umap

import ast
#import the random function library
import random

import os 

from tqdm.auto import tqdm
tqdm.pandas()

train_df = pd.read_csv("/home/gridsan/tmackey/materials_discovery/data/gnome_data/output.csv")

CrystalNN = local_env.CrystalNN(
    distance_cutoffs=None, x_diff_weight=-1, porous_adjustment=False)

# Get the job array index from the environment variable
job_index = int(os.getenv('SLURM_ARRAY_TASK_ID', 0))
total_tasks = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

#job_index = 0
#total_tasks = 20

# Split the dataset into chunks for processing
total_structures = len(train_df)
#divde the total number of structures by the number of tasks and round up
chunk_size = int(np.ceil(total_structures / total_tasks))
start_index = job_index * chunk_size
end_index = min(start_index + chunk_size, total_structures)

print(f"Processing chunk {job_index} of {total_tasks} with {chunk_size} structures")
print(f"Processing structures {start_index} to {end_index}")

def build_crystal(crystal_str, niggli=True, primitive=False):
    """Build crystal from cif string."""
    crystal = Structure.from_str(crystal_str, fmt='cif')

    if primitive:
        crystal = crystal.get_primitive_structure()

    if niggli:
        crystal = crystal.get_reduced_structure()

    canonical_crystal = Structure(
        lattice=Lattice.from_parameters(*crystal.lattice.parameters),
        species=crystal.species,
        coords=crystal.frac_coords,
        coords_are_cartesian=False,
    )
    # match is gaurantteed because cif only uses lattice params & frac_coords
    # assert canonical_crystal.matches(crystal)
    return canonical_crystal


structures = train_df['cif'].iloc[start_index:end_index].progress_apply(build_crystal)

def safe_structure_graph(x):
    try:
        return StructureGraph.with_local_env_strategy(x, CrystalNN)
    except Exception:
        print(x)
        return 0

result_graphs = structures.progress_apply(safe_structure_graph)

