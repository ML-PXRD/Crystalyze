import pandas as pd
import numpy as np
import argparse
import os
import random
import torch

try:
    # Argument parsing
    parser = argparse.ArgumentParser(description='Parallel XRD Simulation')
    parser.add_argument('--n_workers', type=int, help='Total number of workers')
    parser.add_argument('--worker_num', type=int, help='Current worker number (0-indexed)')
    args = parser.parse_args()

    n_workers = args.n_workers
    worker_num = args.worker_num
except:
    n_workers = 1
    worker_num = 0
    
data_dir = '/home/gridsan/tmackey/cdvae/data/mp_20_final/'

train_path = os.path.join(data_dir, 'train_xrd.csv')
val_path = os.path.join(data_dir, 'val_xrd.csv')
test_path = os.path.join(data_dir, 'test_xrd.csv')

train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)
test_df = pd.read_csv(test_path)

dataframes = {'train': train_df, 'val': val_df, 'test': test_df}

for name, data in dataframes.items():
    data['xrd_peak_locations'] = data['xrd_peak_locations'].apply(
        lambda x: [float(i) for i in x.strip('[]').split(',')]
    )

    #make sure we can read in the diffraction patterns
    data['xrd_peak_intensities'] = data['xrd_peak_intensities'].apply(
        lambda x: [float(i) for i in x.strip('[]').split(',')]
    )

def caglioti_fwhm(theta, U, V, W):
    """
    Calculate the FWHM using the Caglioti formula.
    theta: float, the angle in degrees
    U, V, W: Caglioti parameters
    """
    rad_theta = np.radians(theta / 2)  # Convert theta to radians
    return (U * np.tan(rad_theta)**2 + V * np.tan(rad_theta) + W)**0.5

def pseudo_voigt(x, center, amplitude, U, V, W, eta, noise_sd=0.0):
    """
    Pseudo-Voigt function using Caglioti FWHM.
    x: array-like, the independent variable
    center: float, the center of the peak
    amplitude: float, the height of the peak
    U, V, W: Caglioti parameters
    eta: float, the fraction of the Lorentzian component (0 <= eta <= 1)
    """
    fwhm = caglioti_fwhm(center, U, V, W)
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma for Gaussian
    # Generate random noise from a normal distribution
    noise = np.random.normal(0, noise_sd)

    noisy_percentage = (100 + noise_sd) / 100 
    #print("noisy_percentage is ", noisy_percentage)

    #multiply the amplitude by the noisy percentage 
    amplitude = amplitude * noisy_percentage
    
    lorentzian = amplitude * (fwhm**2 / ((x - center)**2 + fwhm**2))
    gaussian = amplitude * np.exp(-(x - center)**2 / (2 * sigma**2))
    return eta * lorentzian + (1 - eta) * gaussian

def superimposed_pseudo_voigt(x, xy_merge, U, V, W, eta, noise_sd=0.0):
    """
    Superimpose multiple pseudo-Voigt functions using Caglioti FWHM.
    x: array-like, the independent variable
    xy_merge: nx2 array, first column is peak locations, second column is intensities
    U, V, W: Caglioti parameters
    eta: float, the fraction of the Lorentzian component (0 <= eta <= 1)
    """
    total = np.zeros_like(x)
    for row in xy_merge:
        center, amplitude = row
        total += pseudo_voigt(x, center, amplitude, U, V, W, eta, noise_sd)
    total = total / max(total)
    return total

# Function to simulate XRD for each row
def simulate_pv_xrd_for_row(row_tuple, U, V, W):
    index, row = row_tuple  # Unpack the tuple

    x = np.arange(5, 90, 0.010)
    eta = 0  # Fraction of Lorentzian component (common for all peaks)

    # Combine peak locations and intensities into a single array
    xy_merge = np.column_stack((row['xrd_peak_locations'], row['xrd_peak_intensities']))

    sim_xrd = superimposed_pseudo_voigt(x, xy_merge, U, V, W, eta, noise_sd=noise)

    return sim_xrd

def apply_simulation(data, U, V, W, worker_num, n_workers, peak_shape = 0):
    # Split data for the current worker
    chunk_size = len(data) // n_workers
    start_idx = worker_num * chunk_size
    end_idx = None if worker_num == n_workers - 1 else start_idx + chunk_size # Last worker gets the rest
    worker_data = data.iloc[start_idx:end_idx]

    print("Worker", worker_num, "processing", len(worker_data), "rows")
    print("start index:", start_idx, "end index:", end_idx)

    # Process using a list comprehension
    results = [simulate_pv_xrd_for_row((idx, row), U, V, W) for idx, row in worker_data.iterrows()]
    
    #turn the list of numpy arrays into a numpy array
    results = np.stack(results)

    tensor = torch.from_numpy(results).float()

    tensor = tensor.reshape(tensor.shape[0], 1, tensor.shape[1])

    data_dict = {}
    for i in range(len(worker_data)):
        key = worker_data['material_id'].iloc[i] + "_" + str(peak_shape)
        data_dict[key] = tensor[i]

    return data_dict

# peak_shapes = [(0.05, -0.06, 0.07), (0.05, -0.01, 0.01),
#                    (0.0, 0.0, 0.01), (0.0, 0.0, random.uniform(0.001, 0.1))]
noise = 0 #this noise corresponds roughly to the effects something like preferred orientation might have on the diffraction pattern. set to 0 for now
peak_shapes = [(0.05, -0.06, 0.07)]

for name, data in dataframes.items():
    # Assuming peak_shapes is defined elsewhere in your code
    for peak_shape, (U, V, W) in enumerate(peak_shapes): 
        sim_pv_xrd_intensities_dict = apply_simulation(data, U, V, W, worker_num, n_workers, peak_shape) #this going to be a n x 8192 array 

        # Save results as a numpy array
        output_filename = f'{name}_sim_pv_xrd_intensities_{peak_shape}_worker_{worker_num}.pt'
        torch.save(sim_pv_xrd_intensities_dict, output_filename)
        print("Saved to {}".format(output_filename))

print("Simulation completed for worker number:", worker_num)