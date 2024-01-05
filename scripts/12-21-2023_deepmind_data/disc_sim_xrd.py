import pandas as pd

import sys
import os
import numpy as np
import ast
from tqdm.auto import tqdm
tqdm.pandas()


#read in the worker number 
try: 
    worker_num = int(sys.argv[1])
except: 
    worker_num = 0

print("Worker number: ", worker_num)

num_splits = 100

try: 
    print('using data_dir', str(sys.argv[2]))
    data_dir = str(sys.argv[2])
except:
    print("using default data_dir /home/gridsan/tmackey/materials_discovery/data/mp_20_for_validation/")
    data_dir = '/home/gridsan/tmackey/materials_discovery/data/mp_20_for_validation/'


#load in the data 
train_df = pd.read_csv(data_dir + 'train_xrd.csv') 
test_df = pd.read_csv(data_dir + 'test_xrd.csv')
#train and test commented out for testing purposes
val_df = pd.read_csv(data_dir + 'val_xrd.csv')
#let's pull out the diffraction patterns ahead of time 
def simulate_xrd(peak_locations, peak_intensities, lower_bound = 5, upper_bound = 75, dimensions = 200):
    interval =  (upper_bound - lower_bound)/dimensions
    sim_positions = np.arange(lower_bound, upper_bound, interval)
    # Create an empty intensity array for the simulation
    sim_intensities = np.zeros_like(sim_positions)

    # Loop over all simulated positions
    for i, pos in enumerate(sim_positions):
        # Find peak locations within 0.25° of the current simulated position
        close_peaks = [(loc, intensity) for loc, intensity in zip(peak_locations, peak_intensities) if abs(loc - pos) <= interval/2]
        
        # If there are close peaks, sum the intensities among those peaks
        if close_peaks:
            intensities = np.array([intensity for loc, intensity in close_peaks])
            sim_intensities[i] = np.sum(intensities)
    
    sim_intensities = 100*sim_intensities / max(sim_intensities)
    
    return sim_intensities

data_frames = {"train": train_df, "test": test_df, "val": val_df}

for name, df in data_frames.items():
    
    chunk_size = np.ceil(len(df)/num_splits)
    
    start_index = int(worker_num*chunk_size)
    end_index = int(min(start_index + chunk_size, len(df))) #prevents end index > len(df)
    sub_df = df.iloc[start_index:end_index].copy()

    print("Processing {} rows".format(len(sub_df)))
    print("Starting index: ", start_index)
    print("Ending index: ", end_index)

    sub_df['xrd_peak_locations'] = sub_df['xrd_peak_locations'].progress_apply(ast.literal_eval)
    sub_df['xrd_peak_intensities'] = sub_df['xrd_peak_intensities'].progress_apply(ast.literal_eval)
    sub_df['disc_sim_xrd'] = sub_df.progress_apply(lambda row: simulate_xrd(row['xrd_peak_locations'], row['xrd_peak_intensities']), axis=1)    

    #save
    sub_df.to_csv(data_dir + f'{name}_xrd_disc_sim_{worker_num}.csv', index=False)