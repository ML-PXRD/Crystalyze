import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from pymatgen.core.structure import Structure, Lattice
from pymatgen.io.cif import CifParser
from io import StringIO

import torch

from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis import local_env

import random

import os 
from tqdm.auto import tqdm
from cdvae.common.data_utils import * 

import sys

import re
import time 
from multiprocessing import Pool

from data_utils.pv_utils import * 

tqdm.pandas()

try: 
    worker_num = int(sys.argv[3])
    num_splits = int(sys.argv[4])
except: 
    worker_num = 0
    num_splits = 1

CrystalNN = local_env.CrystalNN(
    distance_cutoffs=None, x_diff_weight=-1, porous_adjustment=False)

elemental_symbols = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']
OLD_DIRECTORY = False

def generate_structure(elements, stoichiometry):
    """ Generate a random structure with the given elements and stoichiometry.

    Args:
        elements (list): List of elements in the structure. Ex: ['Fe', 'O']
        stoichiometry (list): List of stoichiometry of each element. Ex: [1, 2]

    Returns:
        Structure: A random structure with the given elements and stoichiometry.

    """

    # Generate a random lattice
    a = b = c = random.uniform(3.0, 5.0)  # Random lattice parameters between 3 and 5 Å
    alpha = beta = gamma = 90  # Cubic lattice for simplicity
    lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)

    # Create an empty structure
    structure = Structure(lattice, [], [])

    # Add atoms with random positions
    for element, stoich in zip(elements, stoichiometry):
        for _ in range(stoich):
            # Generate random fractional coordinates
            r_coords = [random.random() for _ in range(3)]
            structure.append(element, r_coords)

    return structure

def generate_cif_string_from_stoichiometry(elements_stoichiometry, rotate_stoichs = False):
    """ Generate a CIF string from a list of elements and their stoichiometry.

    Args:
        elements_stoichiometry (list): List of elements and their stoichiometry. Ex: ['Fe2', 'O3']
        rotate_stoichs (bool): Whether to rotate the stoichiometry to generate multiple structures. Default is False.

    Returns:
        str: A CIF string for the structure with the given elements and stoichiometry.

    """

    if OLD_DIRECTORY:
        # Parse elements and their stoichiometry
        elements, stoichiometry = [], []
        for item in elements_stoichiometry: 
            print(elements_stoichiometry)
            element = ''.join(filter(str.isalpha, item))
            num = ''.join(filter(str.isdigit, item))
        
            stoich = int(num) if num.isdigit() else 1  # Default stoichiometry is 1 if not specified
            elements.append(element)
            stoichiometry.append(stoich)
    else: 
        # Parse elements and their stoichiometry
        elements, stoichiometry = [], []
        index = 0
        while index < len(elements_stoichiometry):
            elements.append(elements_stoichiometry[index])

            if index == len(elements_stoichiometry) - 1: #if we are at the end of the list, set stoiometry to 1 and break
                stoichiometry.append('1')
                break

            if elements_stoichiometry[index+1].isdigit(): #forcast the next value. if it's an int, increment normally
                index += 1
                stoichiometry.append(elements_stoichiometry[index])
                index += 1
            else: # if the stoich is missing, set it to one and keep going
                stoichiometry.append('1')
                index += 1

        stoichiometry = [int(stoich) for stoich in stoichiometry]    
    
    list_of_cifs = []
    if rotate_stoichs:
        #for every integer multiple of the stoichiometry that sums to less than 20
        for i in range(1, 20//sum(stoichiometry) + 1):
            # Generate structure
            structure = generate_structure(elements, [i * stoich for stoich in stoichiometry])

            # Generate CIF string
            cif_string = structure.to(fmt="cif")
            list_of_cifs.append(cif_string)
    else:
        # Generate structure
        structure = generate_structure(elements, [1 * stoich for stoich in stoichiometry])

        # Generate CIF string
        cif_string = structure.to(fmt="cif")
        list_of_cifs.append(cif_string)

    return list_of_cifs

def get_atomic_numbers(cif_data): 
    """ Get the atomic numbers of the atoms in the CIF file.

    Args:
        cif_data (str): The CIF data.

    Returns:
        list: A list of atomic numbers of the atoms in the CIF file.

    """

    cif_file_like_object = StringIO(cif_data)
    parser = CifParser(cif_file_like_object)
    structure = parser.get_structures()[0]
    return list(structure.atomic_numbers)

def get_formula(cif_data): 
    """ Get the formula of the compound in the CIF file.

    Args:
        cif_data (str): The CIF data.

    Returns:
        str: The formula of the compound in the CIF file.

    """

    cif_file_like_object = StringIO(cif_data)
    parser = CifParser(cif_file_like_object)
    structure = parser.get_structures()[0]
    return structure.formula

class XRD_data():
    def __init__(self, data_dir, window_size = 100, s = 0.0001): 
        self.data_dir = data_dir
        self.pv_xrd_dict = self.get_xrds()
        self.window_size = window_size
        self.s = s #s is an adjustable parameter for the spline interpolation. see the docs for more info 
        self.interp_funcs = self.background_subtract_and_interpolate()

    def get_xrds(self): 
        """ 
        Get the XRD data from the data directory.

        Returns:
        dict: A dictionary of the XRD data, with the filename as the key and the XRD data as the value.

        """

        pv_xrd_dictionary = {}
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.csv'):
                csv = pd.read_csv(os.path.join(self.data_dir, filename), header = None, names = ['2theta', 'intensity'])
                pv_xrd_dictionary[filename] = csv

            if filename.endswith('.xy'):
                if "Freedman" in self.data_dir and "CoBi" not in self.data_dir: 
                    csv = pd.read_csv(os.path.join(self.data_dir, filename))
                else:
                    csv = pd.read_csv(os.path.join(self.data_dir, filename), sep='\s+', header=None)
                
                #rename the columns to be '2theta' and 'intensity'
                csv.columns = ['2theta', 'intensity']
                
                pv_xrd_dictionary[filename] = csv

        return pv_xrd_dictionary
    
    def background_subtract_and_interpolate(self, plot = False):
        chunk_size = np.ceil(len(self.pv_xrd_dict)/num_splits)
        
        start_index = int(worker_num*chunk_size)
        end_index = int(min(start_index + chunk_size, len(self.pv_xrd_dict)))

        if start_index > len(self.pv_xrd_dict):
            start_index = len(self.pv_xrd_dict) - 2

        print("start index: ", start_index)
        print("end index: ", end_index)
        print("total length: ", len(self.pv_xrd_dict))

        pv_xrd_dict = {k: self.pv_xrd_dict[k] for k in list(self.pv_xrd_dict.keys())[start_index:end_index]}

        interp_funcs = {}
        for key, pvxrd in tqdm(pv_xrd_dict.items()):
            if plot: 
                plt.plot(pvxrd['2theta'], pvxrd['intensity'] / np.max(pvxrd['intensity']), label = 'original')

            background_subtracted = adaption.background_subtraction(pvxrd['intensity'])
            if plot: 
                plt.plot(pvxrd['2theta'], background_subtracted, label = 'background subtracted')
                plt.show()
            interp_funcs[key] = adaption.interpolate(pvxrd['2theta'], background_subtracted, self.s)

        return interp_funcs
    
    def change_domain(self):
        for key, value in self.pv_xrd_dict.items():
            interp_xrd = self.interp_funcs[key]
            x_new, y_new =  adaption.change_domain(interp_xrd, value['2theta'])
            
            #make a new dataframe 
            new_df = pd.DataFrame({'2theta': x_new, 'intensity': y_new})
            self.pv_xrd_dict[key] = new_df
        
        #save results to the same data directory 
        torch.save(self.pv_xrd_dict, os.path.join(self.data_dir, f"pv_xrd_dict_{worker_num}.pt"))

    
def create_xrd_for_testing(pv_xrd_dict, df):

    """
    Create XRD data for testing.

    Args:
    pv_xrd_dict (dict): A dictionary of the XRD data, with the filename as the key and the XRD data as the value. 
    Ex: {'filename1': DataFrame1, 'filename2': DataFrame2}
    df (DataFrame): A DataFrame of the materials data. This is the same format as train.csv

    Returns:
    dict: A dictionary of the XRD data for testing, with the material_id as the key and the XRD data as the value.
    This can be used directly as the input to the model for testing.

    """

    xrd_for_testing = {} 
    for row in range(df.shape[0]):
        filename = df.iloc[row]['filename']
        material_id = df.iloc[row]['material_id']
        print(filename)
        tensor = torch.tensor(pv_xrd_dict[filename]['intensity'].values)
        normalized_tensor = tensor / torch.max(tensor)
        xrd_for_testing[material_id] = normalized_tensor.unsqueeze(0).float()
        
    return xrd_for_testing

def create_dataframe(dictionary_of_cifs):
    """ 
    This function takes in a dictionary of cifs and filenames and returns a dataframe with the following columns:
    'cif', 'filename', 'formation_energy_per_atom', 'spacegroup.number', 'xrd_peak_intensities', 'xrd_peak_locations', 'disc_sim_xrd', 'atomic_numbers', 'material_id'

    Args:
    dictionary_of_cifs (dict): a dictionary of filenames and their corresponding cifs. Ex: {'filename1': 'cif1', 'filename2': 'cif2'}

    Returns:
    df (pd.DataFrame): a dataframe with the following columns: 'cif', 'filename', 'formation_energy_per_atom', 'spacegroup.number', 'xrd_peak_intensities', 'xrd_peak_locations', 'disc_sim_xrd', 'atomic_numbers', 'material_id'
    """

    #initialize an empty dataframe with 'cif' and 'filename' columns
    df = pd.DataFrame(columns = ['cif', 'filename'])

    #for every key, item pair in the dictionary 
    for key, item in dictionary_of_cifs.items():
        stoich_multiplicity = 1
        for individual_cif in item: 
            #concate the cif and filename to the dataframe
            df = pd.concat([df, pd.DataFrame({'cif': [individual_cif], 'filename': [key], 'material_id': [key.replace(".xy", "") + "_" + str(stoich_multiplicity) + "x"]})])
            stoich_multiplicity += 1
            
    #add the formation_energy_per_atom and spacegroup.number columns and just set them to 0 
    df['formation_energy_per_atom'] = 0
    df['spacegroup.number'] = 0

    #make a 'xrd_peak_intensities' and 'xrd_peak_locations' columns where each entry is 256 * [0]
    df['xrd_peak_intensities'] = [256 * [0] for _ in range(len(df))]
    df['xrd_peak_locations'] = [256 * [0] for _ in range(len(df))]
    df['disc_sim_xrd'] = [np.array(256 * [0]) for _ in range(len(df))]
    df['atomic_numbers'] = df['cif'].apply(get_atomic_numbers)

    return df

def create_xrd_test_data(dir, df):
    """
    Create XRD data for testing.

    Args:
    dir (str): The directory containing the XRD data.

    Returns:
    dict: A dictionary of the XRD data for testing, with the material_id as the key and the XRD data as the value.

    """

    my_XRD_directory = XRD_data(dir)
    my_XRD_directory.change_domain()


def split_elements(filename):
    # Remove the '_raw.xy' part
    clean_name = filename.split('.xy')[0]
    
    # Find all letters and numbers using regular expression
    elements = re.findall(r'[A-Za-z]+|\d+', clean_name)
    
    # Convert numeric strings to integers
    elements = [int(x) if x.isdigit() else x for x in elements]
    
    return elements

def abbrev_to_boolean(str): 
    if str == "Tr": 
        return True
    else:
        return False

def directory_prep(dir):

    """
    This function takes a directory of .xy files and handles edge cases and variable masks by making copies of the .xy files 
    with a different filename.

    Args:

    dir (str): the directory of .xy files

    """
    
    filenames = os.listdir(dir)
    if "Freedman" in dir and ("diffraction_for_Tsach" not in dir):
        for filename in filenames:
            list_of_string_elements = filename[0:filename.find("heat")].split("_")
            extracted_elements = [s for s in list_of_string_elements if s in elemental_symbols]
            
            new_filename = filename
            #catch edge cases 
            if filename in BaFeBi_filenames_original:
                #add "Ba_Fe_Bi" to the beginning of the filename
                new_filename = "Ba_Fe_Bi_" + filename
                os.system("mv " + dir + "/" + filename + " " + dir + "/" + new_filename)
                extracted_elements = ["Ba", "Fe", "Bi"]
            elif filename in KBiO_filenames_original:
                #add "K_Bi_O_heat" to the beginning of the filename
                new_filename = "K_Bi_O_heat" + filename
                os.system("mv " + dir + "/" + filename + " " + dir + "/" + new_filename)
                extracted_elements = ["K", "Bi", "O"]
            elif filename in KBi_filenames_original:
                #replace the filename with K_Bi_O
                new_filename = filename.replace("K_Bi", "K_Bi_O_heat")
                os.system("mv " + dir + "/" + filename + " " + dir + "/" + new_filename)
                extracted_elements = ["K", "Bi", "O"]

            #if the compound is one of K, Bi, O make a copy that has just K and Bi 
            if "K" in extracted_elements and "Bi" in extracted_elements and "O" in extracted_elements:
                new_filename = new_filename.replace("K_Bi_O", "K_Bi")
                os.system("cp " + dir + "/" + filename + " " + dir + "/" + new_filename)

            # if the compound is ['Ba', 'Fe', 'Bi'] make a copy that has ['Ba', 'Fe', 'Bi', 'O']
            if "Ba" in extracted_elements and "Fe" in extracted_elements and "Bi" in extracted_elements:
                new_filename = new_filename.replace("Ba_Fe_Bi_", "Ba_Fe_Bi_O_")
                os.system("cp " + dir + "/" + filename + " " + dir + "/" + new_filename)
    else: 
        print("not Freedman lab data")

hard_coded_name_key_matches = {
    "BaFeBiO": ["Ba", "Fe", "Bi", "O"],
    "RuBi": ["Ru", "Bi"],
    "WBi": ["W", "Bi"],
    "TiBi": ["Ti", "Bi"],
}

def stoichioemtry_extraction(dir):
    """
    This function takes a directory of .xy files and extracts the stoichiometry of the compound from the filename.

    Args:

    dir (str): the directory of .xy files

    Returns: 

    dictionary of cifs (dict): a dictionary of cifs with the filename as the key and the cif string as the value

    """

    dictionary_of_cifs = {}

    filenames = os.listdir(dir)
    
    #need this to handle old file formatting :( 
    if "/home/gridsan/tmackey/cdvae/scripts/XRD_CDVAE/PDF_large_scale/filtered_raw" != dir:
        for filename in filenames:
            print(filename)
            if filename != "temp.ipynb":
                extracted_elements = [element for element in split_elements(filename)]
                rotate_stoichs = extracted_elements[0]
                extracted_elements = extracted_elements[1:]
                
                #sum all the integer elements 
                atom_sum = np.sum([element for element in extracted_elements if type(element) == int])
                
                if atom_sum > 20:
                    continue

                else: 
                    #get the string only elements 
                    string_only = [element for element in extracted_elements if type(element) == str]
                    
                    #check to see if they are all in the elemental symbols list
                    if not all(elem in elemental_symbols for elem in string_only):
                        continue
                    
                    extracted_elements = [str(element) for element in extracted_elements]

                    #cut out the last element, which is jst a marker
                    extracted_elements = extracted_elements[:-1]

                rotate_stoichs = abbrev_to_boolean(rotate_stoichs)
                print(extracted_elements)
                cif_string = generate_cif_string_from_stoichiometry(extracted_elements, rotate_stoichs)
                dictionary_of_cifs[filename] = cif_string

        return dictionary_of_cifs
    else: 
        for filename in filenames:
            print(filename)
            if filename != "temp.ipynb":
                extracted_elements = [element for element in split_elements(filename)]
                #remove anything that is raw
                extracted_elements = [element for element in extracted_elements if element != "raw"]
                rotate_stoichs = extracted_elements[0]
                
                #sum all the integer elements 
                atom_sum = np.sum([element for element in extracted_elements if type(element) == int])
                
                if atom_sum > 20:
                    continue

                else: 
                    #get the string only elements 
                    string_only = [element for element in extracted_elements if type(element) == str]
                    
                    #check to see if they are all in the elemental symbols list
                    if not all(elem in elemental_symbols for elem in string_only):
                        continue
                    
                    extracted_elements = [str(element) for element in extracted_elements]

                rotate_stoichs = abbrev_to_boolean(rotate_stoichs)
                print(extracted_elements)
                cif_string = generate_cif_string_from_stoichiometry(extracted_elements, rotate_stoichs)
                dictionary_of_cifs[filename] = cif_string

        return dictionary_of_cifs

# def stoichioemtry_extraction(dir):
#     """
#     This function takes a directory of .xy files and extracts the stoichiometry of the compound from the filename.

#     Args:

#     dir (str): the directory of .xy files

#     Returns: 

#     dictionary of cifs (dict): a dictionary of cifs with the filename as the key and the cif string as the value

#     """

#     dictionary_of_cifs = {}

#     filenames = os.listdir(dir)

#     for filename in filenames:
#         if ".xy" in filename:
#             if "diffraction_for_Tsach" in dir:
#                 for key, value in hard_coded_name_key_matches.items():
#                     if key in filename:
#                         extracted_elements = value
#                         break
#             elif "Freedman" in dir: 
#                 list_of_string_elements = filename[0:filename.find("heat")].split("_")
#                 extracted_elements = [s for s in list_of_string_elements if s in elemental_symbols]
#             else:
#                 extracted_elements = [element for element in split_elements(filename)]
#                 #sum all the integer elements 
#                 sum = np.sum([element for element in extracted_elements if type(element) == int])
#                 if sum > 20:
#                     continue
#                 else: 
#                     #get the string only elements 
#                     string_only = [element for element in extracted_elements if type(element) == str]
#                     #check to see if they are all in the elemental symbols list
#                     if not all(elem in elemental_symbols for elem in string_only):
#                         continue
#                     extracted_elements = [str(element) for element in extracted_elements]
                        
#             print(filename, extracted_elements)

#             cif_string = generate_cif_string_from_stoichiometry(extracted_elements)
#             dictionary_of_cifs[filename] = cif_string

#     return dictionary_of_cifs

#dictionary_of_cifs = stoichioemtry_extraction("/home/gridsan/tmackey/cdvae/scripts/1-22-2024_clean_impelementations/PDF_large_scale/filtered_raw")
#%%
def build_crystal(crystal_str, niggli=True, primitive=False):
    try: 
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
    except: 
        return None 

def build_crystal_graph(crystal, graph_method='crystalnn'):
    try: 
        """
        """

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
    except: 
        return None

def load_and_merge_graphs(data_dir):
    """
    Args: 
    data_dir: str, the directory where the data is stored

    """
    
    graph_dict = {}
    while len(graph_dict) < num_splits:
        try: 
            for i in range(num_splits):
                graph_dict.update(torch.load(os.path.join(data_dir, f'test_{i}.pt')))
        except: 
            graph_dict = {}
            time.sleep(5)

    #save the dictionary to a file    
    torch.save(graph_dict, os.path.join(data_dir, 'test.pt'))
    
def generate_and_save_graphs(data_dir):
    """
    Args: 
    data_dir: str, the directory where the data is stored

    """
    
    #check if os.path.join(data_dir, 'test.csv') exists (necessary for the 0-(num_splits-1) workers since they got ahead)
    while not os.path.exists(os.path.join(data_dir, 'test.csv')):
        time.sleep(5)

    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    data_frames = {'test': test_df}

    for name, df in data_frames.items():
        
        chunk_size = np.ceil(len(df)/num_splits)
        
        start_index = int(worker_num*chunk_size)
        end_index = int(min(start_index + chunk_size, len(df))) #prevents end index > len(df)
        sub_df = df.iloc[start_index:end_index].copy()
        sub_crystals = sub_df['cif'].progress_apply(build_crystal)
        sub_graphs = sub_crystals.progress_apply(build_crystal_graph)

        materials_ids = sub_df['material_id'].values

        #make a dictionary using the materials_ids as keys and the graphs as values
        graph_dict = dict(zip(materials_ids, sub_graphs))

        #save the dictionary to a file
        torch.save(graph_dict, os.path.join(data_dir, f'test_{worker_num}.pt'))

    if worker_num == num_splits - 1:
        load_and_merge_graphs(data_dir)

def load_across_workers(data_dir, ):
    """
    Load the XRD data across workers.

    Returns:
    dict: A dictionary of the XRD data, with the filename as the key and the XRD data as the value.

    """
    
    pv_xrd_dictionary = {}
    while pv_xrd_dictionary == {}:
        try: 
            for worker_num in range(num_splits):
                pv_xrd_dict = torch.load(os.path.join(data_dir, f"pv_xrd_dict_{worker_num}.pt"))
                pv_xrd_dictionary.update(pv_xrd_dict)
        except:
            #pause for a few seconds and try again 
            time.sleep(5)

    return pv_xrd_dictionary

def main(dir, final_directory, job_type = None):
    """ 
    
    Args:
    dir (str): the directory of .xy files
    final_directory (str): the directory to save the data to

    """

    #directory_prep(dir)
    dictionary_of_cifs = stoichioemtry_extraction(dir)
    df = create_dataframe(dictionary_of_cifs)

    if job_type == "data_prep":
        
        create_xrd_test_data(dir, df)

    elif job_type == "aggregate":
        #to prevent a lot of redundant file accessing, only the last worker will access the files
        
        pv_xrd_dict = load_across_workers(dir)
        print(len(pv_xrd_dict))
        print(df)
        pv_xrd_for_testing = create_xrd_for_testing(pv_xrd_dict, df)

        #make the directory (Ok if it exists)
        try: 
            os.mkdir(final_directory)
        except:
            pass

        #save pv_xrds to /home/gridsan/tmackey/cdvae/data/Freedman_lab_full_subtraction as test_pv_xrd.pt
        torch.save(pv_xrd_for_testing, os.path.join(final_directory, "test_pv_xrd.pt"))

        #save df to /home/gridsan/tmackey/cdvae/data/Freedman_lab_full_subtraction as test.csv
        df.to_csv(os.path.join(final_directory, "test.csv"))

    elif job_type == "graph_prep":
        generate_and_save_graphs(final_directory)

def cif_to_cryslist(dir, filenames):

    """
    Args:
    cif_dir: str, the directory where the cif files are stored

    Returns:
    list: A list of dictionaries, where each dictionary contains the following keys:
    'frac_coords', 'lengths', 'angles', 'atom_types'

    """

    cryslist = []
    for filename in filenames: 
        if "cif" in filename: 
            individual_dict = {}

            structure = Structure.from_file(os.path.join(dir, filename))

            individual_dict['frac_coords'] =  np.array(structure.frac_coords)
            individual_dict['lengths'] = np.array(structure.lattice.lengths)
            individual_dict['angles'] = np.array(structure.lattice.angles)
            individual_dict['atom_types'] = np.array(structure.atomic_numbers)

            cryslist.append(individual_dict)

    return cryslist

def organized_snapped_data(snapped_dir, gt_dir, pre_snap_dir, results_dir): 
    directories = {
        "snapped": snapped_dir,
        "gt": gt_dir,
        "pre_snap": pre_snap_dir,
    }

    snapped_tags = {filename.split("_")[-1] : filename for filename in os.listdir(snapped_dir) if "cif" in filename}
    directories['snapped'] = [cif_to_cryslist(snapped_dir, list(snapped_tags.values()))]
    
    for key, value in zip(["gt", "pre_snap"], [gt_dir, pre_snap_dir]):
        tags = {filename.split("_")[-1] : filename for filename in os.listdir(value) if "cif" in filename}
        directories[key] = [cif_to_cryslist(value, [tags[key] for key in snapped_tags.keys()])]

    for key, value in directories.items():
        torch.save(value, os.path.join(results_dir, f"{key}_crystal_list.pt"))

if __name__ == "__main__":

    dir = sys.argv[1]
    final_directory = sys.argv[2]
    job_type = sys.argv[5]
    
    main(dir, final_directory, job_type)

#Example usage:
# dir = "/home/gridsan/tmackey/cdvae/scripts/XRD_CDVAE/PDF_large_scale/filtered_raw"
# final_directory = "/home/gridsan/tmackey/cdvae/data/PDF_filtered_raw2"
#python data_utils.py /home/gridsan/tmackey/cdvae/scripts/1-22-2024_clean_impelementations/PDF_large_scale/filtered_raw /home/gridsan/tmackey/cdvae/data/PDF_filtered_raw
# %%

###Hopefully this stuff is depcrecated

BaFeBi_filenames_original = ["postheat_minigrid_wide_011_2_BaFeBi_peaks_MgO_BaBi3.xy", 
                    "postheat_minigrid_wide_011_2_BaFeBi_peaks_MgO.xy",
                    "postheat_minigrid_wide_011_2_BaFeBiO_peaks_MgO_BaBi3.xy",
                    "postheat_minigrid_wide_011_2_BaFeBi_raw_MgO_BaBi3.xy",
                    "postheat_minigrid_wide_011_2_BaFeBi_raw_MgO.xy",
                    "postheat_minigrid_wide_011_2_BaFeBi_raw_BaBi3.xy",
                    "postheat_minigrid_wide_011_2_BaFeBi_peaks_BaBi3.xy",
                    "postheat_minigrid_wide_011_2_BaFeBi_raw_MgO_BaBi3_new.xy"]

KBiO_filenames_original = ["K_Bi_O_BiV_3_7887_MgO41495_peaks_MgO_Bi_V.xy",
                           "K_Bi_O_BiV_3_7887_MgO41495_peaks_MgO.xy",
                           "K_Bi_O_BiV_3_7887_MgO41495_peaks_Bi_V.xy",
                           "K_Bi_O_postheat_054_peaks_MgO.xy"]

KBi_filenames_original = ["K_Bi_BiV_3_7887_MgO41495_peaks_MgO_Bi_V.xy",
                          "K_Bi_O_BiV_3_7887_MgO41495_raw_Bi_V.xy",
                          "K_Bi_O_BiV_3_7887_MgO41495_raw_MgO.xy",
                          "K_Bi_O_BiV_3_7887_MgO41495_raw_MgO_Bi_V.xy"]