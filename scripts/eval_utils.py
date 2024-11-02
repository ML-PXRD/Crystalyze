import itertools
import numpy as np
import torch
import hydra

from scipy.spatial.distance import pdist
from scipy.spatial.distance import cdist
from hydra.experimental import compose
from hydra import initialize_config_dir
from pathlib import Path

import smact
from smact.screening import pauling_test

from cdvae.common.constants import CompScalerMeans, CompScalerStds
from cdvae.common.data_utils import StandardScaler, chemical_symbols

import os 
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from collections import Counter
from pymatgen.core.composition import Composition
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.analysis.diffraction.xrd import XRDCalculator
xrd_calculator = XRDCalculator(wavelength='CuKa', symprec=0.1)

import matplotlib.pyplot as plt
from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer

from tqdm import tqdm

CompScaler = StandardScaler(
    means=np.array(CompScalerMeans),
    stds=np.array(CompScalerStds),
    replace_nan_token=0.)

def load_data(file_path):
    if file_path[-3:] == 'npy':
        data = np.load(file_path, allow_pickle=True).item()
        for k, v in data.items():
            if k == 'input_data_batch':
                for k1, v1 in data[k].items():
                    data[k][k1] = torch.from_numpy(v1)
            else:
                data[k] = torch.from_numpy(v).unsqueeze(0)
    else:
        data = torch.load(file_path, map_location=torch.device('cpu'))
    return data


def get_model_path(eval_model_name):
    import cdvae
    model_path = (
        Path(cdvae.__file__).parent / 'prop_models' / eval_model_name)
    return model_path


def load_config(model_path):
    with initialize_config_dir(str(model_path)):
        cfg = compose(config_name='hparams')
    return cfg


def load_model(model_path, load_data=False, testing=True, test_set_override=None):
    with initialize_config_dir(str(model_path)):
        if test_set_override is not None:
            cfg = compose(config_name='hparams', overrides=[f"data.root_path=/home/gridsan/tmackey/cdvae/data/{test_set_override}",
                                                            f"data.eval_model_name={test_set_override}"])
            print("overriding data with ", test_set_override)
        else:
            cfg = compose(config_name='hparams')
        model = hydra.utils.instantiate(
            cfg.model,
            optim=cfg.optim,
            data=cfg.data,
            logging=cfg.logging,
            _recursive_=False,
        )
        ckpts = list(model_path.glob('*.ckpt'))
        if len(ckpts) > 0:
            ckpt_epochs = np.array(
                [int(ckpt.parts[-1].split('-')[0].split('=')[1]) for ckpt in ckpts])
            ckpt = str(ckpts[ckpt_epochs.argsort()[-1]])
        model = model.load_from_checkpoint(ckpt, strict=False)
        model.lattice_scaler = torch.load(model_path / 'lattice_scaler.pt')
        model.scaler = torch.load(model_path / 'prop_scaler.pt')

        if load_data:
            datamodule = hydra.utils.instantiate(
                cfg.data.datamodule, _recursive_=False, scaler_path=model_path
            )
            if testing:
                datamodule.setup('test')
                print(datamodule)
                print(datamodule.test_dataloader())
                test_loader = datamodule.test_dataloader()[0]
                print(test_loader)
            else:
                datamodule.setup()
                test_loader = datamodule.val_dataloader()[0]
        else:
            test_loader = None

    return model, test_loader, cfg

def smact_validity(comp, count,
                   use_pauling_test=True,
                   include_alloys=True):
    elem_symbols = tuple([chemical_symbols[elem] for elem in comp])
    space = smact.element_dictionary(elem_symbols)
    smact_elems = [e[1] for e in space.items()]
    electronegs = [e.pauling_eneg for e in smact_elems]
    ox_combos = [e.oxidation_states for e in smact_elems]
    if len(set(elem_symbols)) == 1:
        return True
    if include_alloys:
        is_metal_list = [elem_s in smact.metals for elem_s in elem_symbols]
        if all(is_metal_list):
            return True

    threshold = np.max(count)
    compositions = []
    for ox_states in itertools.product(*ox_combos):
        stoichs = [(c,) for c in count]
        # Test for charge balance
        cn_e, cn_r = smact.neutral_ratios(
            ox_states, stoichs=stoichs, threshold=threshold)
        # Electronegativity test
        if cn_e:
            if use_pauling_test:
                try:
                    electroneg_OK = pauling_test(ox_states, electronegs)
                except TypeError:
                    # if no electronegativity data, assume it is okay
                    electroneg_OK = True
            else:
                electroneg_OK = True
            if electroneg_OK:
                for ratio in cn_r:
                    compositions.append(
                        tuple([elem_symbols, ox_states, ratio]))
    compositions = [(i[0], i[2]) for i in compositions]
    compositions = list(set(compositions))
    if len(compositions) > 0:
        return True
    else:
        return False

def structure_validity(crystal, cutoff=0.5):
    dist_mat = crystal.distance_matrix
    # Pad diagonal with a large number
    dist_mat = dist_mat + np.diag(
        np.ones(dist_mat.shape[0]) * (cutoff + 10.))
    if dist_mat.min() < cutoff or crystal.volume < 0.1:
        return False
    else:
        return True

def get_fp_pdist(fp_array):
    if isinstance(fp_array, list):
        fp_array = np.array(fp_array)
    fp_pdists = pdist(fp_array)
    return fp_pdists.mean()

def filter_fps(struc_fps, comp_fps):
    assert len(struc_fps) == len(comp_fps)

    filtered_struc_fps, filtered_comp_fps = [], []

    for struc_fp, comp_fp in zip(struc_fps, comp_fps):
        if struc_fp is not None and comp_fp is not None:
            filtered_struc_fps.append(struc_fp)
            filtered_comp_fps.append(comp_fp)
    return filtered_struc_fps, filtered_comp_fps

def compute_cov(crys, gt_crys,
                struc_cutoff, comp_cutoff, num_gen_crystals=None):
    struc_fps = [c.struct_fp for c in crys]
    comp_fps = [c.comp_fp for c in crys]
    gt_struc_fps = [c.struct_fp for c in gt_crys]
    gt_comp_fps = [c.comp_fp for c in gt_crys]

    assert len(struc_fps) == len(comp_fps)
    assert len(gt_struc_fps) == len(gt_comp_fps)

    # Use number of crystal before filtering to compute COV
    if num_gen_crystals is None:
        num_gen_crystals = len(struc_fps)

    struc_fps, comp_fps = filter_fps(struc_fps, comp_fps)

    comp_fps = CompScaler.transform(comp_fps)
    gt_comp_fps = CompScaler.transform(gt_comp_fps)

    struc_fps = np.array(struc_fps)
    gt_struc_fps = np.array(gt_struc_fps)
    comp_fps = np.array(comp_fps)
    gt_comp_fps = np.array(gt_comp_fps)

    struc_pdist = cdist(struc_fps, gt_struc_fps)
    comp_pdist = cdist(comp_fps, gt_comp_fps)

    struc_recall_dist = struc_pdist.min(axis=0)
    struc_precision_dist = struc_pdist.min(axis=1)
    comp_recall_dist = comp_pdist.min(axis=0)
    comp_precision_dist = comp_pdist.min(axis=1)

    cov_recall = np.mean(np.logical_and(
        struc_recall_dist <= struc_cutoff,
        comp_recall_dist <= comp_cutoff))
    cov_precision = np.sum(np.logical_and(
        struc_precision_dist <= struc_cutoff,
        comp_precision_dist <= comp_cutoff)) / num_gen_crystals

    metrics_dict = {
        'cov_recall': cov_recall,
        'cov_precision': cov_precision,
        'amsd_recall': np.mean(struc_recall_dist),
        'amsd_precision': np.mean(struc_precision_dist),
        'amcd_recall': np.mean(comp_recall_dist),
        'amcd_precision': np.mean(comp_precision_dist),
    }

    combined_dist_dict = {
        'struc_recall_dist': struc_recall_dist.tolist(),
        'struc_precision_dist': struc_precision_dist.tolist(),
        'comp_recall_dist': comp_recall_dist.tolist(),
        'comp_precision_dist': comp_precision_dist.tolist(),
    }

    return metrics_dict, combined_dist_dict

def get_file_paths(root_path, task, label='', suffix='pt'):
   if label == '':
       out_name = f'eval_{task}.{suffix}'
   else:
       out_name = f'eval_{task}_{label}.{suffix}'
   out_name = os.path.join(root_path, out_name)
   return out_name

def get_crystals_list(
       frac_coords, atom_types, lengths, angles, num_atoms):
   """
   args:
       frac_coords: (num_atoms, 3)
       atom_types: (num_atoms)
       lengths: (num_crystals)
       angles: (num_crystals)
       num_atoms: (num_crystals)
   """
   assert frac_coords.size(0) == atom_types.size(0) == num_atoms.sum()
   assert lengths.size(0) == angles.size(0) == num_atoms.size(0)


   start_idx = 0
   crystal_array_list = []
   for batch_idx, num_atom in enumerate(num_atoms.tolist()):
       cur_frac_coords = frac_coords.narrow(0, start_idx, num_atom)
       cur_atom_types = atom_types.narrow(0, start_idx, num_atom)
       cur_lengths = lengths[batch_idx]
       cur_angles = angles[batch_idx]


       crystal_array_list.append({
           'frac_coords': cur_frac_coords.detach().cpu().numpy(),
           'atom_types': cur_atom_types.detach().cpu().numpy(),
           'lengths': cur_lengths.detach().cpu().numpy(),
           'angles': cur_angles.detach().cpu().numpy(),
       })
       start_idx = start_idx + num_atom
   return crystal_array_list

class Crystal(object):
   def __init__(self, crys_array_dict):
       self.frac_coords = crys_array_dict['frac_coords']
       self.atom_types = crys_array_dict['atom_types']
       self.lengths = crys_array_dict['lengths']
       self.angles = crys_array_dict['angles']
       self.dict = crys_array_dict

       self.get_structure()
       self.get_composition()
       self.get_validity()
       #self.get_fingerprints()
   def get_structure(self):
       if min(self.lengths.tolist()) < 0:
           self.constructed = False
           self.invalid_reason = 'non_positive_lattice'
       else:
           try:
               self.structure = Structure(
                   lattice=Lattice.from_parameters(
                       *(self.lengths.tolist() + self.angles.tolist())),
                   species=self.atom_types, coords=self.frac_coords, coords_are_cartesian=False)
               self.constructed = True
           except Exception:
               self.constructed = False
               self.invalid_reason = 'construction_raises_exception'
           if self.structure.volume < 0.1:
               self.constructed = False
               self.invalid_reason = 'unrealistically_small_lattice'
   def get_composition(self):
       elem_counter = Counter(self.atom_types)
       composition = [(elem, elem_counter[elem])
                      for elem in sorted(elem_counter.keys())]
       elems, counts = list(zip(*composition))
       counts = np.array(counts)
       counts = counts / np.gcd.reduce(counts)
       self.elems = elems
       self.comps = tuple(counts.astype('int').tolist())
   def get_validity(self):
       self.comp_valid = smact_validity(self.elems, self.comps)
       if self.constructed:
           self.struct_valid = structure_validity(self.structure)
       else:
           self.struct_valid = False
       self.valid = self.comp_valid and self.struct_valid
   def get_fingerprints(self):
       elem_counter = Counter(self.atom_types)
       comp = Composition(elem_counter)
       self.comp_fp = CompFP.featurize(comp)
       try:
           site_fps = [CrystalNNFP.featurize(
               self.structure, i) for i in range(len(self.structure))]
       except Exception:
           # counts crystal as invalid if fingerprint cannot be constructed.
           print('oops')
           self.valid = False
           self.comp_fp = None
           self.struct_fp = None
           return
       self.struct_fp = np.array(site_fps).mean(axis=0)
      
class RecEval(object):
   def __init__(self, pred_crys, gt_crys, stol=0.5, angle_tol=10, ltol=0.3): #original values of stol=0.5, angle_tol=10, ltol=0.3
       assert len(pred_crys) == len(gt_crys)
       self.matcher = StructureMatcher(
           stol=stol, angle_tol=angle_tol, ltol=ltol)
       self.preds = pred_crys
       self.gts = gt_crys

   def get_match_rate_and_rms(self):
       def process_one(pred, gt, is_valid):
           if not is_valid:
               return None
           try:
               rms_dist = self.matcher.get_rms_dist(
                   pred.structure, gt.structure)
               rms_dist = None if rms_dist is None else rms_dist[0]
               return rms_dist
           except Exception:
               return None
       #define a function that gets the diffraction patterns for pred and gt and returns the RMSD between them
       def process_diff_pattern(pred, gt, is_valid):
           if not is_valid:
               return None
           try:
               #get the structures
               pred_structure = pred.structure
               gt_structure = gt.structure
               pred_pattern = xrd_calculator.get_pattern(pred_structure)
               gt_pattern = xrd_calculator.get_pattern(gt_structure)


               pred_adjusted_vector = np.zeros(256)
               minimum = min(256, len(pred_pattern.x))
               pred_adjusted_vector[:minimum] = pred_pattern.x[:minimum]


               gt_adjusted_vector = np.zeros(256)
               minimum = min(256, len(gt_pattern.x))
               gt_adjusted_vector[:minimum] = gt_pattern.x[:minimum]
              
               #calculate the RMSD between the two patterns
               print(pred_adjusted_vector)
               print(gt_adjusted_vector)
               rms_dist = np.sqrt(np.mean((pred_adjusted_vector - gt_adjusted_vector)**2))


               return rms_dist
           except Exception:
               return None   


       validity = [c.valid for c in self.preds]

       rms_dists = []
       evaluate_diff_pattern = False
       if evaluate_diff_pattern:
           diff_dists = []
       for i in range(len(self.preds)):
           rms_dists.append(process_one(
               self.preds[i], self.gts[i], validity[i]))
           if evaluate_diff_pattern:
               diff_dists.append(process_diff_pattern(self.preds[i], self.gts[i], validity[i]))
       rms_dists = np.array(rms_dists)
       if evaluate_diff_pattern:
           diff_dists = np.array(diff_dists)
           average_diff_dist = diff_dists[diff_dists != None].mean()
           #print out all the diff dists
       else:
           average_diff_dist = None
       match_rate = sum(rms_dists != None) / len(self.preds)
       mean_rms_dist = rms_dists[rms_dists != None].mean()

       return {'match_rate': match_rate,
               'rms_dist': mean_rms_dist,
               'diff_dist': average_diff_dist,
               'rmsd_values': rms_dists}

   def get_metrics(self):
       return self.get_match_rate_and_rms()
   

def count_unique_crystals(pred_crys):
    is_unique_list = []
    for i in range(len(pred_crys) - 1):
        is_unique = True
        #determine if they are the sume 
        rec_evaluator = RecEval((len(pred_crys) - (i+1)) * [pred_crys[i]], pred_crys[(i + 1): len(pred_crys)])
        recon_metrics = rec_evaluator.get_metrics()
        numeric_metrics = np.array([0 if x is None else x for x in recon_metrics['rmsd_values']])

        #recon metrics rmsd_values will be greater than 0 if they are the same 
        #if the indices are different and the rmsd is none 
        if np.sum(numeric_metrics) != 0:
            is_unique = False
                
        is_unique_list.append(is_unique)
    
    is_unique_list.append(True)

    return is_unique_list

def get_unique_crystals(unique_list, pred_crystals): 
    return [pred_crystals[i] for i in range(len(pred_crystals)) if unique_list[i]]

def symmetryops(structure, symprec):
    sga = SpacegroupAnalyzer(structure, symprec=symprec)
    space_group_symbol = sga.get_space_group_symbol()
    space_group_number = sga.get_space_group_number()  # Get the space group number
    symmetrized_structure = sga.get_refined_structure()
    
    return space_group_symbol, space_group_number, symmetrized_structure

def xrd_plotter(structure1, structure2 = None, xlim = None): 
    # Calculate the XRD pattern
    pattern = xrd_calculator.get_pattern(structure1)
    
    if structure2: 
        pattern2 = xrd_calculator.get_pattern(structure2)

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.vlines(pattern.x, 0, pattern.y, colors='blue', lw=0.5)
    if structure2: 
        plt.vlines(pattern2.x, 0, pattern2.y, colors='red', lw=0.5)
    plt.xlabel('2θ [degrees]')
    plt.ylabel('Intensity [arbitrary units]')
    plt.title('X-ray Diffraction Pattern')
    if xlim:
        plt.xlim(xlim[0], xlim[1]) 
    plt.grid()
    plt.show()


def get_correct_crystals(metrics, crys): 
    return [crys[i] for i in range(len(crys)) if metrics[i]]

def get_crystal_array_list(file_path, batch_idx=0):
    data = load_data(file_path)
    crys_array_list = get_crystals_list(
        data['frac_coords'][batch_idx],
        data['atom_types'][batch_idx],
        data['lengths'][batch_idx],
        data['angles'][batch_idx],
        data['num_atoms'][batch_idx])

    if 'input_data_batch' in data:
        batch = data['input_data_batch']
        if isinstance(batch, dict):
            true_crystal_array_list = get_crystals_list(
                batch['frac_coords'], batch['atom_types'], batch['lengths'],
                batch['angles'], batch['num_atoms'])
        else:
            true_crystal_array_list = get_crystals_list(
                batch.frac_coords, batch.atom_types, batch.lengths,
                batch.angles, batch.num_atoms)
    else:
        true_crystal_array_list = None
    try: 
        predicted_property = data['predicted_property'][batch_idx]
    except Exception as e: 
        predicted_property = None

    return crys_array_list, true_crystal_array_list, predicted_property

def all_results_retreival(recon_file_path, num_evals = 1, label = ""):
   """
   
   Get the results for all batches in the recon file path 
   
   """

   all_results = []
   all_gt = []

   for eval_num in tqdm(range(num_evals)): #for every batch 
       file_path = recon_file_path

       if eval_num > 0:
            #get the filepath 
            if label == "":
                file_path = file_path[:-3]+ "__{}.pt".format(eval_num)
            else:
                file_path = file_path[:-3]+ "_{}.pt".format(eval_num)

            crys_array_list, true_crystal_array_list, predicted_properties = get_crystal_array_list(file_path)
       
       else:
            
            crys_array_list, true_crystal_array_list, predicted_properties = get_crystal_array_list(file_path)
        
       all_results.append(crys_array_list) #this will have dimensions (num_evals, num_crystals)
       all_gt.append(true_crystal_array_list) #this will have dimensions (num_evals, num_crystals)

   return all_results, all_gt

def is_unique(list_of_structures, structure):
   try:
       results = RecEval(list_of_structures, len(list_of_structures) * [structure]).get_metrics()
   except:
       results = {'rmsd_values': len(list_of_structures) * [None]}
   results_with_0 = [0 if x is None else x for x in results['rmsd_values']]
   if np.sum(results_with_0) > 0:
       return False
   else:
       return True

def composition_checker(all_results_crystals, all_gt_crystals):
    composition_matches = []
    for crystal_index in range(len(all_results_crystals)):

        predicted_composition = all_results_crystals[crystal_index].get_composition()
        gt_composition = all_gt_crystals[crystal_index].get_composition()

        composition_matches.append(predicted_composition == gt_composition)

    return composition_matches

def evaluation(all_results, all_gt, num_evals = 1, stol=0.5, angle_tol=10, ltol=0.3):
    #Get the rmsd values for all the results in all_results and all_gt with the given tolerances

    total_rmsd = []
    total_rmsd_just_sites = []

    for eval_indx in tqdm(range(num_evals)):
        all_results_crystals = [Crystal(x) for x in all_results[eval_indx]]
        all_gt_crystals = [Crystal(x) for x in all_gt[eval_indx]]

        rec_evaluator = RecEval(all_results_crystals, all_gt_crystals, stol=stol, angle_tol=angle_tol, ltol=ltol)
        
        rec_evaluator_just_sites = composition_checker(all_results_crystals, all_gt_crystals)
        
        try:
            recon_metrics = rec_evaluator.get_metrics()

        except Exception as e:
            print('error is ', e)
            recon_metrics = {'rmsd_values': len(all_gt[eval_indx]) * [None]}
        
        total_rmsd.append(recon_metrics['rmsd_values'])    
        total_rmsd_just_sites.append(rec_evaluator_just_sites)
        
        
    total_rmsd = np.array(total_rmsd)
    total_rmsd_stacked = np.stack(total_rmsd)

    total_rmsd_just_sites = np.array(total_rmsd_just_sites)
    total_rmsd_just_sites_stacked = np.stack(total_rmsd_just_sites)

    return total_rmsd_stacked, total_rmsd_just_sites_stacked

def tolerance_analysis(prediction, ground_truth, ltol_values, angle_tol_values, num_evals):

    """ 
    
    Args: 

    prediction: list of torch tensors, each tensor is a batch of predicted structures
    ground_truth: list of torch tensors, each tensor is a batch of ground truth structures

    Returns:

    evals_per_ltol: numpy array of shape (len(ltol_values), 3), each row is the evaluation results for a given ltol value
    evals_per_angle_tol: numpy array of shape (len(angle_tol_values), 3), each row is the evaluation results for a given angle_tol value

    """

    evals_per_ltol = []
    for ltol in tqdm(ltol_values):
        evals_per_ltol.append(evaluation(prediction, ground_truth, num_evals = num_evals, ltol=ltol)[0])

    evals_per_angle_tol = []
    for angle_tol in tqdm(angle_tol_values):
        evals_per_angle_tol.append(evaluation(prediction, ground_truth, num_evals = num_evals, angle_tol=angle_tol)[0])

    evals_per_ltol = np.stack(evals_per_ltol)
    evals_per_angle_tol = np.stack(evals_per_angle_tol)

    return evals_per_ltol, evals_per_angle_tol


def load_and_merge(data_dir, label, total_data = None, num_workers = 100): 
    for worker in range(num_workers): 
        
        filename =  'worker_' + str(worker) + "_" + label + '.npy'
        
        if worker == 0: 
            total_data = np.load(data_dir + '/' + filename)
        
        else:
            total_data_chunk = np.load(data_dir + '/' + filename)
        
            if len(total_data_chunk.shape) == 3: #this deals with the tolerance case. 

                total_data = np.concatenate((total_data, total_data_chunk), axis = 2)
            
            else:
            
                total_data = np.concatenate((total_data, total_data_chunk), axis = 1)

    return total_data

def get_file_paths(root_path, task, label='', suffix='pt'):
    if label == '':
        out_name = f'eval_{task}.{suffix}'
    else:
        out_name = f'eval_{task}_{label}.{suffix}'
    out_name = os.path.join(root_path, out_name)
    return out_name
