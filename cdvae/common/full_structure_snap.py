import cdvae.common.diffraction as dc
import torch
from pymatgen.core import Structure
from pymatgen.core import Lattice
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import torch.nn.functional as Fun
from cdvae.common.segfault_protect import segfault_protect
from cdvae.common.instantiate_sga import instantiate_spacegroup_analyzer
import multiprocessing




if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')


class FullStructureSnapper(dc.diffraction_pattern):
    def __init__(self, q_min = 0.5, q_max = 0.2, wavelength = 1.5406, in_dim = 8500, output_dim = 256, scn = False, parameter_dictionaries = None, plot_dictionary = None):
        super().__init__(q_min = q_min, q_max = q_max, wavelength = wavelength, in_dim = in_dim, output_dim = output_dim, scn = scn)
        if plot_dictionary is None:
            self.plot_dictionary = {'plot_progress': False,
                                'plot_freq': 10,
                                'graph_losses': False}
        else:
            self.plot_dictionary = plot_dictionary
        
        if parameter_dictionaries is None:
            self.parameter_dictionaries = [
                {'batch_size': 16,
                'bin_batch_size': 8,
                'learning_rate': 0.0004,
                'iterations': 30001,
                'UVW': torch.tensor([0,0,1.5], device='cuda:0'),
                'snap_lengths': True,
                'snap_angles': False,
                'snap_coords': False,
                'snap_UVW': [False, False, True],
                'symmeterize': True,
                'force_angles': False,
                'slow_peak_shape': False,
                'convergence_tolerance': -0.001,
                'UVW_exponent': 1.5,
                'UVW_rate_slow': 1,
                'UVW_rate': 5,
                'finished': False
                },
                {'batch_size': 16,
                'bin_batch_size': 8,
                'learning_rate': 0.0004,
                'iterations': 30001,
                'UVW': torch.tensor([0,0,1.5], device='cuda:0'),
                'snap_lengths': True,
                'snap_angles': False,
                'snap_coords': True,
                'snap_UVW': [False, False, True],
                'symmeterize': True,
                'force_angles': False,
                'slow_peak_shape': False,
                'convergence_tolerance': -0.001,
                'UVW_exponent': 1.5,
                'UVW_rate_slow': 1,
                'UVW_rate': 5,
                'finished': True
                },
                {'batch_size': 16,
                'bin_batch_size': 8,
                'learning_rate': 0.04,
                'iterations': 30001,
                'UVW': torch.tensor([0,0,3.0], device='cuda:0'),
                'snap_lengths': True,
                'snap_angles': True,
                'snap_coords': False,
                'snap_UVW': [False, False, False],
                'symmeterize': False,
                'force_angles': False,
                'slow_peak_shape': False,
                'convergence_tolerance': -0.001,
                'UVW_exponent': 1.5,
                'UVW_rate_slow': 1,
                'UVW_rate': 5,
                'finished': False
                },
                {'batch_size': 16,
                'bin_batch_size': 8,
                'learning_rate': 0.0004,
                'iterations': 30001,
                'UVW': torch.tensor([0,0,1.5], device='cuda:0'),
                'snap_lengths': True,
                'snap_angles': True,
                'snap_coords': True,
                'snap_UVW': [False, False, True],
                'symmeterize': False,
                'force_angles': False,
                'slow_peak_shape': False,
                'convergence_tolerance': -0.001,
                'UVW_exponent': 1.5,
                'UVW_rate_slow': 1,
                'UVW_rate': 5,
                'finished': True
                },
                {'batch_size': 16,
                'bin_batch_size': 8,
                'learning_rate': 0.04,
                'iterations': 30001,
                'UVW': torch.tensor([0,0,30.0], device='cuda:0'),
                'snap_lengths': True,
                'snap_angles': True,
                'snap_coords': False,
                'snap_UVW': [False, False, False],
                'symmeterize': False,
                'force_angles': True,
                'slow_peak_shape': False,
                'convergence_tolerance': -0.00001,
                'UVW_exponent': 1.5,
                'UVW_rate_slow': 1,
                'UVW_rate': 5,
                'finished': False
                },
                {'batch_size': 16,
                'bin_batch_size': 8,
                'learning_rate': 0.0004,
                'iterations': 30001,
                'UVW': torch.tensor([0,0,3.0], device='cuda:0'),
                'snap_lengths': False,
                'snap_angles': False,
                'snap_coords': True,
                'snap_UVW': [False, False, False],
                'symmeterize': False,
                'force_angles': False,
                'slow_peak_shape': False,
                'convergence_tolerance': -0.00001,
                'UVW_exponent': 1.5,
                'UVW_rate_slow': 1,
                'UVW_rate': 5,
                'finished': False
                },
                {'batch_size': 16,
                'bin_batch_size': 8,
                'learning_rate': 0.0004,
                'iterations': 30001,
                'UVW': torch.tensor([0,0,3.0], device='cuda:0'),
                'snap_lengths': True,
                'snap_angles': True,
                'snap_coords': False,
                'snap_UVW': [False, False, False],
                'symmeterize': True,
                'force_angles': False,
                'slow_peak_shape': False,
                'convergence_tolerance': -0.001,
                'UVW_exponent': 1.5,
                'UVW_rate_slow': 1,
                'UVW_rate': 5,
                'finished': False
                },
                {'batch_size': 16,
                'bin_batch_size': 8,
                'learning_rate': 0.0004,
                'iterations': 30001,
                'UVW': torch.tensor([0,0,1.5], device='cuda:0'),
                'snap_lengths': True,
                'snap_angles': True,
                'snap_coords': True,
                'snap_UVW': [False, False, True],
                'symmeterize': False,
                'force_angles': False,
                'slow_peak_shape': False,
                'convergence_tolerance': -0.001,
                'UVW_exponent': 1.5,
                'UVW_rate_slow': 1,
                'UVW_rate': 5,
                'finished': True
                },
                {'batch_size': 16,
                'bin_batch_size': 8,
                'learning_rate': 0.04,
                'iterations': 30001,
                'UVW': torch.tensor([0,0,30.0], device='cuda:0'),
                'snap_lengths': True,
                'snap_angles': True,
                'snap_coords': False,
                'snap_UVW': [False, False, False],
                'symmeterize': False,
                'force_angles': True,
                'slow_peak_shape': False,
                'convergence_tolerance': -0.00001,
                'UVW_exponent': 1.5,
                'UVW_rate_slow': 1,
                'UVW_rate': 5,
                'finished': False
                },
                {'batch_size': 16,
                'bin_batch_size': 8,
                'learning_rate': 0.0004,
                'iterations': 30001,
                'UVW': torch.tensor([0,0,3.0], device='cuda:0'),
                'snap_lengths': False,
                'snap_angles': False,
                'snap_coords': True,
                'snap_UVW': [False, False, False],
                'symmeterize': False,
                'force_angles': False,
                'slow_peak_shape': False,
                'convergence_tolerance': -0.00001,
                'UVW_exponent': 1.5,
                'UVW_rate_slow': 1,
                'UVW_rate': 5,
                'finished': False
                },
                {'batch_size': 16,
                'bin_batch_size': 8,
                'learning_rate': 0.0004,
                'iterations': 30001,
                'UVW': torch.tensor([0,0,3.0], device='cuda:0'),
                'snap_lengths': True,
                'snap_angles': True,
                'snap_coords': False,
                'snap_UVW': [False, False, False],
                'symmeterize': False,
                'force_angles': False,
                'slow_peak_shape': False,
                'convergence_tolerance': -0.001,
                'UVW_exponent': 1.5,
                'UVW_rate_slow': 1,
                'UVW_rate': 5,
                'finished': False
                },
                {'batch_size': 16,
                'bin_batch_size': 8,
                'learning_rate': 0.0004,
                'iterations': 30001,
                'UVW': torch.tensor([0,0,1.5], device='cuda:0'),
                'snap_lengths': True,
                'snap_angles': True,
                'snap_coords': True,
                'snap_UVW': [False, False, True],
                'symmeterize': False,
                'force_angles': False,
                'slow_peak_shape': False,
                'convergence_tolerance': -0.001,
                'UVW_exponent': 1.5,
                'UVW_rate_slow': 1,
                'UVW_rate': 5,
                'finished': True
                }
                ]
        else:
            self.parameter_dictionaries = parameter_dictionaries
    
    def structure_to_batch(self, structure):
        batch = {
                'lengths': torch.tensor([[structure.lattice.a, structure.lattice.b, structure.lattice.c]], dtype= torch.float32, device='cuda:0'),
                'angles': torch.tensor([[structure.lattice.alpha, structure.lattice.beta, structure.lattice.gamma]], dtype= torch.float32, device='cuda:0'),
                'frac_coords': torch.tensor([site.frac_coords for site in structure], dtype = torch.float32, device='cuda:0'),
                'num_atoms': torch.tensor([structure.num_sites], device='cuda:0'),
                'atom_types': torch.tensor([site.specie.number for site in structure], device='cuda:0')
            }
        return batch

    def batch_to_structure(self, batch):
        lattice = Lattice.from_parameters(a=batch['lengths'][0][0].item(), b=batch['lengths'][0][1].item(), c=batch['lengths'][0][2].item(), alpha=batch['angles'][0][0].item(), beta=batch['angles'][0][1].item(), gamma=batch['angles'][0][2].item())
        structure = Structure(lattice, species=batch['atom_types'].cpu().numpy(), coords=batch['frac_coords'].cpu().detach().numpy(), coords_are_cartesian=False)
        return structure


    # Use segfault_protect to protect against common segfaults
    def find_sym(self, snapped_batch, tolerance = 2.6, angle_tolerance = 5):
        structure = self.batch_to_structure(snapped_batch)
        unsnapped = True
        while unsnapped:
            try:
                analyzer = segfault_protect(instantiate_spacegroup_analyzer, structure, symprec=tolerance, angle_tolerance=angle_tolerance)
                refined_structure = analyzer.get_refined_structure()
                unsnapped = False
            except:
                if tolerance > 0.01:
                    tolerance -= 0.5
                    print('Symmetry finder failed, decreasing tolerance to', tolerance)
                else:
                    print('Symmetry finder failed')
                    refined_structure = structure
                    unsnapped = False
        snapped_batch = self.structure_to_batch(refined_structure)
        return snapped_batch


    def full_structure_snap(self, patterns, initial_structures):
        if patterns.size()[0] < self.parameter_dictionaries[0]['batch_size']:
            self.parameter_dictionaries[0]['batch_size'] = patterns.size()[0]
        if patterns.size()[0] < self.parameter_dictionaries[0]['bin_batch_size']:
            self.parameter_dictionaries[0]['bin_batch_size'] = patterns.size()[0]

        mini_batch_list = self.split_batch(initial_structures, self.parameter_dictionaries[0]['batch_size'])
        patterns = self.background_subtraction(patterns)
        patterns_list = []
        for i in range(int(patterns.size(0)/self.parameter_dictionaries[0]['batch_size'])):
            patterns_list.append(patterns[i*self.parameter_dictionaries[0]['batch_size']:(i+1)*self.parameter_dictionaries[0]['batch_size']])
        snapped_lengths = []
        snapped_angles = []
        snapped_coords = []
        combined_loss_list = []

        for mini_batch_index in range(len(mini_batch_list)):
            snapped_batch = mini_batch_list[mini_batch_index]
            patterns = patterns_list[mini_batch_index]

            #Save the initial input structures and patterns
            final_loss_list = []
            refined_snapped_structure_list = []
            initial_structure_list = []
            initial_batch_list = self.split_batch(snapped_batch, 1)
            for batch in initial_batch_list:
                initial_structure_list.append(self.batch_to_structure(batch))
            initial_patterns = patterns.clone()
            finished_batch_list = []

            # Do all the refinements described in self.parameter_dictionary
            for parameter_dictionary in self.parameter_dictionaries:
                # Check if all structures are finished
                if len(initial_structure_list) - len(finished_batch_list) == 0:
                    break

                current_parameter_dictionary = parameter_dictionary.copy()
                current_parameter_dictionary['batch_size'] = parameter_dictionary['batch_size'] - len(finished_batch_list)
                if current_parameter_dictionary['bin_batch_size'] > current_parameter_dictionary['batch_size']:
                    current_parameter_dictionary['bin_batch_size'] = current_parameter_dictionary['batch_size']

                # For the angles to higher symmetry
                if current_parameter_dictionary['force_angles'] == True:
                    with torch.no_grad():
                        for j in range(snapped_batch['angles'].size()[0]):
                            for i in range(snapped_batch['angles'].size()[1]):
                                if snapped_batch['angles'][j][i] > 110 and snapped_batch['angles'][j][i] < 130:
                                    snapped_batch['angles'][j][i] = 120
                                if snapped_batch['angles'][j][i] > 80 and snapped_batch['angles'][j][i] < 100:
                                    snapped_batch['angles'][j][i] = 90
                                if snapped_batch['angles'][j][i] > 50 and snapped_batch['angles'][j][i] < 70:
                                    snapped_batch['angles'][j][i] = 60

                # Symmeterize the structures
                if current_parameter_dictionary['symmeterize'] == True:
                    snapped_batch_list = self.split_batch(snapped_batch, 1)
                    symmeterized_snapped_batch_list = []
                    for snapped_batch in snapped_batch_list:
                        snapped_batch = self.find_sym(snapped_batch)
                        symmeterized_snapped_batch_list.append(snapped_batch)
                    snapped_batch = self.merge_batch(symmeterized_snapped_batch_list)

                # Perform the refinement
                snapped_batch, UVW, final_losses = self.fine_snap_structure(patterns, snapped_batch, parameter_dictionary = current_parameter_dictionary, plot_dictionary = self.plot_dictionary)
                
                #Put the finished structures back into the list, record metrics, then remove finished structures again
                if current_parameter_dictionary['finished'] == True:
                    snapped_batch_list = self.split_batch(snapped_batch, 1)
                    snapped_structure_list = []
                    finished_counter = 0
                    filled_final_losses = []
                    for i in range(len(initial_structure_list)):
                        if i in finished_batch_list:
                            snapped_structure_list.append(initial_structure_list[i])
                            filled_final_losses.append(0)
                            finished_counter += 1
                        else:
                            snapped_structure_list.append(self.batch_to_structure(snapped_batch_list[i - finished_counter]))
                            filled_final_losses.append(final_losses[i - finished_counter])
                    
                    final_loss_list.append(filled_final_losses)
                    refined_snapped_structure_list.append(snapped_structure_list)
                
                    for i in range(len(filled_final_losses)):
                        if filled_final_losses[i] < -0.95:
                            finished_batch_list.append(i)
                    
                    # Remove finished structures
                    snapped_batch_list = []
                    for i in range(len(initial_structure_list)):
                        if i not in finished_batch_list:
                            snapped_batch_list.append(self.structure_to_batch(initial_structure_list[i]))
                    if len(snapped_batch_list) > 0:
                        snapped_batch = self.merge_batch(snapped_batch_list)

                    # Remove finished patterns
                    pattern_mask = torch.ones(initial_patterns.size(0), dtype=torch.bool)
                    pattern_mask[finished_batch_list] = False
                    patterns = initial_patterns[pattern_mask]



            # Collect the best refinement for every structure
            final_loss_array = np.array(final_loss_list)
            snapped_batch_list = []
            print(self.parameter_dictionaries)
            print(final_loss_array.shape)
            for i in range(final_loss_array.shape[1]):
                snapped_batch_list.append(self.structure_to_batch(refined_snapped_structure_list[np.argmin(final_loss_array[:,i])][i]))
                combined_loss_list.append(np.min(final_loss_array[:,i]))
            snapped_batch = self.merge_batch(snapped_batch_list)
        
            snapped_lengths.append(snapped_batch['lengths'])
            snapped_angles.append(snapped_batch['angles'])
            snapped_coords.append(snapped_batch['frac_coords'])
        
        snapped_batch['lengths'] = torch.cat(snapped_lengths, dim = 0)
        snapped_batch['angles'] = torch.cat(snapped_angles, dim = 0)
        snapped_batch['frac_coords'] = torch.cat(snapped_coords, dim = 0)


        return snapped_batch, combined_loss_list
    


    def fine_snap_structure(self, pattern, initial_structure, parameter_dictionary = None, plot_dictionary = None):
        if parameter_dictionary is None:
            parameter_dictionary = {'batch_size': 16,
                                    'bin_batch_size': 8,
                                    'learning_rate': 0.0004,
                                    'iterations': 30001,
                                    'UVW': torch.tensor([0,0,1.5], device='cuda:0'),
                                    'snap_lengths': True,
                                    'snap_angles': False,
                                    'snap_coords': True,
                                    'snap_UVW': [False, False, True],
                                    'symmeterize': True,
                                    'force_angles': False,
                                    'slow_peak_shape': False,
                                    'convergence_tolerance': -0.001,
                                    'UVW_exponent': 1.5,
                                    'UVW_rate_slow': 1,
                                    'UVW_rate': 5,
                                    'finished': True
                                    }
        
        if plot_dictionary is None:
            plot_dictionary = {'plot_progress': False,
                               'plot_freq': 10,
                               'graph_losses': True}

        losses = [[0] * (parameter_dictionary['batch_size'] + 1)]
        pattern = pattern / torch.max(pattern, 1).values.unsqueeze(1)
        initial_structure['lengths'].requires_grad_(True)
        initial_structure['angles'].requires_grad_(True)
        initial_structure['frac_coords'].requires_grad_(True)
        if parameter_dictionary['UVW'] == None:
            parameter_dictionary['UVW'] = torch.tensor([[0,0,1.0]], device='cuda:0')
            parameter_dictionary['UVW'] = parameter_dictionary['UVW'].repeat(pattern.size()[0], 1)
        elif len(parameter_dictionary['UVW'].size()) == 1:
            parameter_dictionary['UVW'] = parameter_dictionary['UVW'].repeat(pattern.size()[0], 1)
        parameter_dictionary['UVW'].requires_grad_(True)

        # Gradient descent function using PyTorch autograd
        i = 0
        finished = False
        best_loss = [0] * parameter_dictionary['batch_size']
        lengths_update = 0
        angles_update = 0
        coords_update = 0
        best_UVW = torch.zeros(parameter_dictionary['UVW'].size()[0], 3, device='cuda:0')
        best_structure_list = [0] * parameter_dictionary['batch_size']
        U_update = torch.zeros(parameter_dictionary['UVW'].size()[0], device='cuda:0')
        V_update = torch.zeros(parameter_dictionary['UVW'].size()[0], device='cuda:0')
        W_update = torch.zeros(parameter_dictionary['UVW'].size()[0], device='cuda:0')
        lengths_update = torch.zeros(initial_structure['lengths'].size()[0], 3, device='cuda:0')
        angles_update = torch.zeros(initial_structure['angles'].size()[0], 3, device='cuda:0')
        coords_update = torch.zeros(initial_structure['frac_coords'].size()[0], 3, device='cuda:0')
        W_slow_mask = torch.zeros(parameter_dictionary['UVW'].size()[0], dtype = torch.bool, device='cuda:0')
        

        while finished == False and i < parameter_dictionary['iterations']:
            i = i + 1
            loss_list, simulated_pattern = self.calculate_fine_loss(pattern, initial_structure, UVW = parameter_dictionary['UVW'], mini_batch_size = parameter_dictionary['batch_size'], bin_batch_size = parameter_dictionary['bin_batch_size'])

            loss_list[-1].backward()
            loss_list[-1] = loss_list[-1].item()
            loss_mask = loss_list > losses[-1] and losses[-1] < best_loss
            with torch.no_grad():
                parameter_dictionary['UVW'][:,0] += U_update * loss_mask
                parameter_dictionary['UVW'][:,1] += V_update * loss_mask
                parameter_dictionary['UVW'][:,2] += W_update * loss_mask
                initial_structure['lengths'] += lengths_update * loss_mask
                initial_structure['angles'] += angles_update * loss_mask
                initial_structure['frac_coords'] += coords_update * loss_mask

            split_batch_list = self.split_batch(initial_structure, 1)
            for j in range(len(loss_list)-1):
                if loss_list[j] > losses[-1][j] and losses[-1][j] < best_loss[j]:
                    best_structure_list[j] = self.batch_to_structure(split_batch_list[j])
                    best_loss[j] = losses[-1][j]
            
            with torch.no_grad():
                parameter_dictionary['UVW'][j,0] -= U_update[j] * loss_mask
                parameter_dictionary['UVW'][j,1] -= V_update[j] * loss_mask
                parameter_dictionary['UVW'][j,2] -= W_update[j] * loss_mask
                initial_structure['lengths'][j] -= lengths_update[j] * loss_mask
                initial_structure['angles'][j] -= angles_update[j] * loss_mask
                initial_structure['frac_coords'][j] -= coords_update[j] * loss_mask

            losses.append(loss_list)


            with torch.no_grad():

                if parameter_dictionary['slow_peak_shape']:
                    parameter_dictionary['UVW'].grad = parameter_dictionary['UVW'].grad * parameter_dictionary['UVW_rate_slow'] * (parameter_dictionary['UVW'][:,2]**parameter_dictionary['UVW_exponent']).unsqueeze(1)
                else:
                    parameter_dictionary['UVW'].grad = parameter_dictionary['UVW'].grad * parameter_dictionary['UVW_rate'] * (parameter_dictionary['UVW'][:,2]**parameter_dictionary['UVW_exponent']).unsqueeze(1)
                if parameter_dictionary['snap_UVW'][0]:
                    #U_mask = (UVW[:,0] - learning_rate * UVW.grad[:,0] < 0) or ((UVW[:,0] - learning_rate * UVW.grad[:,0]).isnan())
                    U_update = parameter_dictionary['learning_rate'] * parameter_dictionary['UVW'].grad[:,0]
                    U_update[torch.isnan(U_update)] = 0
                    parameter_dictionary['UVW'][:,0] -= U_update

                if parameter_dictionary['snap_UVW'][1]:
                    #V_mask = (UVW[:,1] - learning_rate * UVW.grad[:,1] < 0) or ((UVW[:,1] - learning_rate * UVW.grad[:,1]).isnan())
                    V_update = parameter_dictionary['learning_rate'] * parameter_dictionary['UVW'].grad[:,1]
                    V_update[torch.isnan(V_update)] = 0
                    parameter_dictionary['UVW'][:,1] -= V_update
                
                if parameter_dictionary['snap_UVW'][2]:
                    #W_mask = (UVW[:,2] - learning_rate * UVW.grad[:,2] < 0) or ((UVW[:,2] - learning_rate * UVW.grad[:,2]).isnan())
                    W_slow_mask = (torch.abs(parameter_dictionary['learning_rate'] * parameter_dictionary['UVW'].grad[:,2]) > 0.1 * parameter_dictionary['UVW'][:,2])
                    W_update = W_slow_mask.int() * 0.1 * torch.abs(parameter_dictionary['UVW'][:,2]) * (parameter_dictionary['UVW'].grad[:,2] / torch.abs(parameter_dictionary['UVW'].grad[:,2]))
                    W_update += (~W_slow_mask).int() * parameter_dictionary['learning_rate'] * parameter_dictionary['UVW'].grad[:,2]
                    W_update[torch.isnan(W_update)] = 0
                    parameter_dictionary['UVW'][:,2] -= W_update
                parameter_dictionary['UVW'].grad.zero_()



                if parameter_dictionary['snap_lengths']:
                    length_slow_mask = (torch.abs(parameter_dictionary['learning_rate'] * initial_structure['lengths'].grad) > 0.1 * initial_structure['lengths'])
                    lengths_update = length_slow_mask.int() * 0.01 * initial_structure['lengths'] * (initial_structure['lengths'].grad / torch.abs(initial_structure['lengths'].grad))
                    lengths_update += (~length_slow_mask).int() * parameter_dictionary['learning_rate'] * initial_structure['lengths'].grad
                    lengths_update[torch.isnan(lengths_update)] = 0
                    initial_structure['lengths'] -= lengths_update
                initial_structure['lengths'].grad.zero_() # Clear the gradients for the next 



                if parameter_dictionary['snap_angles']:
                    angles_slow_mask = (torch.abs(parameter_dictionary['learning_rate'] * initial_structure['angles'].grad) > 0.1 * initial_structure['angles'])
                    angles_update = angles_slow_mask.int() * 0.01 * initial_structure['angles'] * (initial_structure['angles'].grad / torch.abs(initial_structure['angles'].grad))
                    angles_update += (~angles_slow_mask).int() * parameter_dictionary['learning_rate'] * initial_structure['angles'].grad
                    angles_update[torch.isnan(angles_update)] = 0
                    initial_structure['angles'] -= angles_update
                initial_structure['angles'].grad.zero_() # Clear the gradients for the next 



                if parameter_dictionary['snap_coords']:
                    coords_slow_mask = (torch.abs(parameter_dictionary['learning_rate'] * initial_structure['frac_coords'].grad) > 0.1)
                    coords_update = coords_slow_mask.int() * 0.001 * initial_structure['frac_coords'] * (initial_structure['frac_coords'].grad / torch.abs(initial_structure['frac_coords'].grad))
                    coords_update += (~coords_slow_mask).int() * parameter_dictionary['learning_rate'] * initial_structure['frac_coords'].grad
                    coords_update[torch.isnan(coords_update)] = 0
                    initial_structure['frac_coords'] -= coords_update
                initial_structure['frac_coords'].grad.zero_() # Clear the gradients for the next

                try:
                    if (losses[-1][-1] - losses[-1000][-1] > parameter_dictionary['convergence_tolerance']):
                        finished = True
                except:
                    pass

            if plot_dictionary['plot_progress'] and i % plot_dictionary['plot_freq'] == 0:
                start_2theta = np.arcsin((self.q_min * self.wavelength) / (4 * pi)) * 360 / pi
                stop_2theta = np.arcsin((self.q_max * self.wavelength) / (4 * pi)) * 360 / pi
                step_size = (stop_2theta - start_2theta) / simulated_pattern.size()[1]

                simulated_domain = np.arange(simulated_pattern.size()[1]) * step_size + start_2theta
                self.plot_progress(simulated_domain, simulated_pattern, pattern)
                print(f"Step {i+1}: Lengths - {initial_structure['lengths']}, Angles - {initial_structure['angles']}, Frac Coords - {initial_structure['frac_coords']}, UVW - {parameter_dictionary['UVW']}, Losses - {losses[-1]}")
        
        snapped_batch = initial_structure
        final_losses = losses[-1]

        snapped_batch_list = self.split_batch(snapped_batch, 1)
        with torch.no_grad():
            for j in range(len(final_losses)-1):
                if final_losses[j] > best_loss[j]:
                    final_losses[j] = best_loss[j]
                    snapped_batch_list[j] = self.structure_to_batch(best_structure_list[j])
                    parameter_dictionary['UVW'][j] = best_UVW[j]
            snapped_batch = self.merge_batch(snapped_batch_list)

        #print(final_losses)
        
        # Free up memory
        del simulated_pattern, pattern, initial_structure, loss_list, best_loss, best_UVW, best_structure_list


        if plot_dictionary['graph_losses']:
            losses_array = np.array(losses)
            for i in range(losses_array.shape[1]):
                plt.plot(losses_array[:,i])
                plt.show()
                plt.close()
        return snapped_batch, parameter_dictionary['UVW'], final_losses
    

    def calculate_fine_loss(self, patterns, structures, mini_batch_size = 8, bin_batch_size = 8, UVW = None, q_min = None, q_max = None):
        if q_min is not None:
            self.q_min = q_min
        if q_max is not None:
            self.q_max = q_max
        if UVW == None:
            UVW = torch.tensor([[0,0,1.0]], requires_grad = True, device='cuda:0')
            UVW = UVW.repeat(patterns.size()[0], 1)
        num_steps = patterns.size()[1]
        binned_initial_struct_pattern = self.batch_split_bin_pattern_theta(self.batch_split_diffraction_calc(structures, mini_batch_size), bin_batch_size, UVW = UVW, num_steps = num_steps)
        cosine_sim = -1*Fun.cosine_similarity(patterns, binned_initial_struct_pattern, dim = 1)
        loss_list = cosine_sim.tolist()
        loss_list.append(torch.sum(-1*Fun.cosine_similarity(patterns, binned_initial_struct_pattern, dim = 1)))
        return loss_list, binned_initial_struct_pattern



    def plot_progress(self, simulated_domain, simulated_pattern, patterns):
        simulated_pattern = simulated_pattern / torch.max(simulated_pattern, 1).values.unsqueeze(1)
        patterns = patterns / torch.max(patterns, 1).values.unsqueeze(1)
        for i in range(simulated_pattern.size()[0]):
            plt.plot(simulated_domain, patterns[i].cpu(), label = 'Experimental Pattern')
            plt.plot(simulated_domain, simulated_pattern[i].cpu().detach().numpy(), label = 'Simulated Pattern')
            plt.show()
            plt.close()
    
    def adam_tuner(self, patterns, initial_structures, UVW, plot_freq = 1000):
        optimizer = torch.optim.Adam([initial_structures['lengths'], initial_structures['angles'], initial_structures['frac_coords'], UVW], lr = 0.003)
        for step in range(100000):
            optimizer.zero_grad()
            loss_list, simulated_pattern = self.calculate_fine_loss(patterns, initial_structures, UVW = UVW)
            loss = loss_list[-1]
            loss.backward()
            optimizer.step()
            if step % plot_freq == 0:
                start_2theta = np.arcsin((self.q_min * self.wavelength) / (4 * pi)) * 360 / pi
                stop_2theta = np.arcsin((self.q_max * self.wavelength) / (4 * pi)) * 360 / pi
                step_size = (stop_2theta - start_2theta) / simulated_pattern.size()[1]

                simulated_domain = np.arange(simulated_pattern.size()[1]) * step_size + start_2theta
                self.plot_progress(simulated_domain, simulated_pattern, patterns)
                print(f"Step {step+1}: Lengths - {initial_structures['lengths']}, Angles - {initial_structures['angles']}, Frac Coords - {initial_structures['frac_coords']}, UVW - {UVW}, Losses - {loss}")
        return initial_structures
    
