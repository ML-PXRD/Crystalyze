import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch 
from evaluate_utils.eval_utils import *

import pandas as pd

class result_data: 
    def __init__(self, path, num_evals, num_batches = None): 
        self.total_rmsd = load_and_merge(path, label = 'fs_total_rmsd', num_workers = 100)
        self.cumulative_total_rmsd_per_eval = self.get_per_eval_results(self.total_rmsd)

        self.ltols = load_and_merge(path, label = 'evals_per_ltol', num_workers = 100)
        self.atols = load_and_merge(path, label = 'evals_per_angle_tol', num_workers = 100)
        self.snapped_ltols = load_and_merge(path, label = 'snapped_evals_per_ltol', num_workers = 100)
        self.snapped_atols = load_and_merge(path, label = 'snapped_evals_per_angle_tol', num_workers = 100)

        self.per_batch_results = []
        if num_batches: 
            self.get_per_batch_results(num_batches)

        self.ltols_per_tol = self.get_per_eval_results_for_tol(self.ltols)
        self.atols_per_tol = self.get_per_eval_results_for_tol(self.atols)
        self.snapped_ltols_per_tol = self.get_per_eval_results_for_tol(self.snapped_ltols)
        self.snapped_atols_per_tol = self.get_per_eval_results_for_tol(self.snapped_atols)

        self.label = path.split('/')[-1]
        #get the path before the label
        self.model_path = get_file_paths(path.split(self.label)[0], 'recon', label = self.label)

        self.pred, self.gt = self.load_crystals(self.model_path, self.label, num_evals)

    def get_per_eval_results(self, data): 
        data_no_nans = np.nan_to_num(data, nan = 0)
        data_cumulative_sum = np.cumsum(data_no_nans, axis = 0)
        data_cumulative_sum_per_eval = np.mean(data_cumulative_sum > 0, axis = 1)
        return data_cumulative_sum_per_eval
    
    def get_per_eval_results_for_tol(self, data):
        return np.stack([self.get_per_eval_results(data[i]) for i in range(len(data))])
    
    def load_crystals(self, model_path, label, num_evals):
        return all_results_retreival(model_path, num_batches = num_evals, label = label)

    def get_per_batch_results(self, num_batches): 
        
        for i in range(num_batches):
            indices = [i * 256, (i + 1) * 256]
            one_batch_result = self.total_rmsd[:, indices[0]:indices[1]]
            self.per_batch_results.append(self.get_per_eval_results(one_batch_result))

def comparing_total_to_5batch(total_batch_results, five_batch_results, axs):
    x_values = np.arange(4)
    y_values_five_batch = five_batch_results.cumulative_total_rmsd_per_eval[0:4]
    y_values_total_batch = total_batch_results.cumulative_total_rmsd_per_eval[0:4]

    axs.plot(x_values, y_values_five_batch, color='blueviolet')
    axs.plot(x_values, y_values_total_batch, color='dodgerblue')
    axs.legend(['5 Batch Results', 'Total Batch Results'], fontsize=12)

    scatter_y_values = []
    for i in range(5):
        one_batch_result = five_batch_results.per_batch_results[i]
        scatter_y_values.append(one_batch_result[0:4])
        axs.scatter(x_values, one_batch_result[0:4], color='blueviolet')

    axs.set_title('Simulated Test Data: Match Rate vs Number of Attempts')
    axs.set_xlabel('Number of Attempts', fontsize=12)
    axs.set_ylabel('Match Rate', fontsize=12)
    axs.set_xticks(range(0, 4, 1))
    axs.tick_params(labelsize=12)

    df = pd.DataFrame({
        "Number of attempts": x_values,
        "Match rate 5 batch": y_values_five_batch,
        "Match rate total batch": y_values_total_batch,
        "Match rate batch 0": scatter_y_values[0],
        "Match rate batch 1": scatter_y_values[1],
        "Match rate batch 2": scatter_y_values[2],
        "Match rate batch 3": scatter_y_values[3],
        "Match rate batch 4": scatter_y_values[4]
    })

    return df

def getting_5batch_results_to_max(five_batch_results, axs):
    x_values = np.arange(64)
    y_values = five_batch_results.cumulative_total_rmsd_per_eval[0:64]

    axs.plot(x_values, y_values, color='blueviolet')
    axs.legend(['5 Batch Results'], fontsize=12)

    scatter_y_values = []
    for i in range(5):

        one_batch_result = five_batch_results.per_batch_results[i]
        scatter_y_values.append(one_batch_result)
        axs.scatter(x_values, one_batch_result, color='blueviolet')

    axs.set_title('Simulated Test Data: Match Rate vs Number of Attempts')
    axs.set_xlabel('Number of Attempts', fontsize=12)
    axs.set_ylabel('Match Rate', fontsize=12)
    axs.set_xticks(range(0, 64, 16))
    axs.tick_params(labelsize=12)

    df = pd.DataFrame({
        'Number of attempts': x_values,
        'Match rate': y_values,
        "Match rate batch 0": scatter_y_values[0],
        "Match rate batch 1": scatter_y_values[1],
        "Match rate batch 2": scatter_y_values[2],
        "Match rate batch 3": scatter_y_values[3],
        "Match rate batch 4": scatter_y_values[4]
    })

    return df

def augmentation_results(non_augmented_model_results, augmented_model_results, axs):
    x_values = np.arange(64)
    y_values_non_augmented = non_augmented_model_results.cumulative_total_rmsd_per_eval
    y_values_augmented = augmented_model_results.cumulative_total_rmsd_per_eval

    axs.plot(x_values, y_values_non_augmented, color='crimson')
    axs.plot(x_values, y_values_augmented, color='orangered')
    axs.legend(['Non-Augmented Model', 'Augmented Model'], fontsize=12)

    axs.set_title('RRUFF Test Data: Match Rate vs Number of Attempts')
    axs.set_xlabel('Number of Attempts', fontsize=12)
    axs.set_ylabel('Match Rate', fontsize=12)
    axs.set_xticks(range(0, 64, 16))
    axs.tick_params(labelsize=12)

    df = pd.DataFrame({
        'Number of attempts': x_values,
        'Match rate non-augmented': y_values_non_augmented,
        'Match rate augmented': y_values_augmented
    })

    return df

def first_evaluation_figure(full_test_set_results, five_batch_results, RRUFF_non_aug_model, RRUFF_aug_model):
    fig, axs = plt.subplots(1, 3, figsize=(20, 8))

    dataframes = (comparing_total_to_5batch(full_test_set_results, five_batch_results, axs[0]),
                    getting_5batch_results_to_max(five_batch_results, axs[1]),
                    augmentation_results(RRUFF_non_aug_model, RRUFF_aug_model, axs[2])) 
   
    return fig, dataframes

class plot_xrds():
    def plot(pattern, x, y):
            #normalize the y values
            pattern.y = pattern.y / pattern.y.max()

            # Plotting
            plt.figure(figsize=(10, 4))
            plt.stem(pattern.x, pattern.y)
            
            plt.plot(x, y, color='red', alpha=0.5)
            plt.xlabel('2θ (Degrees)')
            plt.ylabel('Intensity')
            plt.title('XRD Pattern')
            plt.show()