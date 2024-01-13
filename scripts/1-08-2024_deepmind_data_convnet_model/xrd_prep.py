from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.diffraction.xrd import XRDCalculator

import pandas as pd

import sys
import os
import numpy as np
#read in the worker number 
try: 
    worker_num = int(sys.argv[1])
except: 
    worker_num = 0

num_splits = 100

print("worker_num", worker_num)
print("num_splits", num_splits)

try: 
    print('using data_dir', str(sys.argv[2]))
    data_dir = str(sys.argv[2])
except:
    print("using default data_dir /home/gridsan/tmackey/cdvae/scripts/1-06-2024_augmentation/proper_data_pipeline_data")
    data_dir = '/home/gridsan/tmackey/cdvae/scripts/1-06-2024_augmentation/proper_data_pipeline_data'

#load in the data 
train_df = pd.read_csv(data_dir + 'train.csv')
test_df = pd.read_csv(data_dir + 'test.csv')
val_df = pd.read_csv(data_dir + 'val.csv')

# Initialize the XRDCalculator with a wavelength of CuKa (1.54060 Å)
xrd_calculator = XRDCalculator(wavelength='CuKa')
from tqdm.auto import tqdm
tqdm.pandas()

def get_xrd_information(crystal_str):
    try: 
        crystal = Structure.from_str(crystal_str, fmt='cif')
    except:
        crystal = None

    try:  
        xrd = xrd_calculator.get_pattern(crystal)
    except: 
        xrd = None

    try: 
        x = xrd.x.tolist()
        y = xrd.y.tolist()
    except:
        x = None
        y = None

    try: 
        atomic_species = [Element(specie).Z for specie in crystal.species]
    except: 
        atomic_species = None

    return [xrd, x, y, atomic_species]

data_frames = {"train": train_df, "test": test_df, "val": val_df}

for name, df in data_frames.items():
    
    chunk_size = np.ceil(len(df)/num_splits)
    
    start_index = int(worker_num*chunk_size)
    end_index = int(min(start_index + chunk_size, len(df))) #prevents end index > len(df)
    
    print("start_index", start_index)
    print("end_index", end_index)

    sub_df = df.iloc[start_index:end_index].copy()
    sub_crystals = sub_df['cif'].progress_apply(get_xrd_information)
    sub_df['xrd'] = sub_crystals.progress_apply(lambda x: x[0])
    sub_df['xrd_peak_locations'] = sub_crystals.progress_apply(lambda x: x[1])
    sub_df['xrd_peak_intensities'] = sub_crystals.progress_apply(lambda x: x[2])
    sub_df['atomic_numbers'] = sub_crystals.progress_apply(lambda x: x[3])

    #save the csv
    sub_df.to_csv(data_dir + f'{name}_xrd_{worker_num}.csv', index=False)