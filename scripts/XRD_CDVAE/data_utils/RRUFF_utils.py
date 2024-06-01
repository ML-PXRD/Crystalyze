import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from tqdm import tqdm 

from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifParser
from pymatgen.io.cif import CifWriter

from io import StringIO
import datetime

from pymatgen.core import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator
import matplotlib.pyplot as plt

from pymatgen.core.periodic_table import Element

from data_utils.pv_utils import * 
from viz_utils.viz import *
from data_utils.structure_utils import *

class RRUFF_data():
    """
    Class for extracting and processing data from a directory input of RRUFF data.
    For reference: xy_data_dir: "/home/gridsan/tmackey/cdvae/scripts/1-13-2024_RUFF_data/Combined/XY_Processed"
    cif_data_dir: "/home/gridsan/tmackey/cdvae/scripts/1-13-2024_RUFF_data/mydir/"    
    """

    def __init__(self, xy_data_dir, cif_data_dir, raw_text = None, xrd_df = None, cif_dict = None, cif_df = None, label = None, filter_inversion = False):
        self.label = label
        self.now = datetime.datetime.now()
        self.dirname = os.path.join("ckpt", self.label, self.now.strftime("%Y-%m-%d"))
        os.makedirs(self.dirname, exist_ok=True)

        self.xy_data_dir = xy_data_dir 
        self.cif_data_dir = cif_data_dir

        self.raw_text = raw_text
        self.xrd_df = xrd_df
        self.cif_dict = cif_dict
        self.cif_df = cif_df

        extractor = extraction(self.dirname, self.xy_data_dir, self.cif_data_dir)
        processor = processing(self.dirname)

        if raw_text is None and xrd_df is None:
            self.raw_text = extractor.read_in_data()

        if xrd_df is None: 
            self.xrd_df = extractor.create_xrd_dataframe(self.raw_text)
            self.xrd_df = processor.remove_calculated_patterns(self.xrd_df)

        if cif_dict is None and cif_df is None:
            self.cif_dict = extractor.create_cif_dict(self.xrd_df)
            self.cif_dict = extractor.convert_cif_strings_to_structures(self.cif_dict)
            
        if cif_df is None: 
            self.cif_dict = processor.filter_structures(self.cif_dict, filter_inversion)
            self.cif_dict = processor.remove_duplicates(self.cif_dict)
            self.create_csv()
        
        self.reformat_csv()
   
    def create_csv(self):
        taskname = "filtered_df"

        # Constructing the DataFrame directly from the dictionary
        self.cif_df = pd.DataFrame({
            'cif': [value['structure'] for value in self.cif_dict.values()],
            'material_id': [value['name'] for value in self.cif_dict.values()],
        })

        # Save the DataFrame and the list of XRDs as checkpoints
        checkpointing.save_checkpoint(self.dirname, taskname, self.cif_df, "csv")

    def structure_to_cif_string(self, structure):
        """
        Convert a pymatgen structure object to a CIF string.
        """
        cif_writer = CifWriter(structure)
        return cif_writer.__str__()

    def reformat_csv(self):
        """
        Reformat the dataframe to have the cif strings as strings instead of pymatgen structure objects.
        """

        # Apply the conversion function to all structures in the dataframe
        self.cif_df['cif'] = self.cif_df['cif'].apply(self.structure_to_cif_string)

        # Save the reformatted DataFrame and the pv_dict as checkpoints
        checkpointing.save_checkpoint(self.dirname, "reformatted_df", self.cif_df, "csv")

    def tensorize_xrds(pv_dict):
        pv_dict_2 = pv_dict
        for key in pv_dict_2.keys():
            pv_dict_2[key] = torch.tensor(pv_dict_2[key])
            #unsqueeze all the values to be 1x8500
            pv_dict_2[key] = pv_dict_2[key].unsqueeze(0)
            #change the type of the values to be floats 
            pv_dict_2[key] = pv_dict_2[key].type(torch.FloatTensor)
            #normalize the values by the max value
            pv_dict_2[key][0][:, 1] = pv_dict_2[key][0][:, 1] / pv_dict_2[key][0][:, 1].max()

        return pv_dict_2

class extraction():
    """
    Suite of functions for extracting and processing data from a directory input of RRUFF data.
    """

    def __init__(self, dirname, xy_data_dir, cif_data_dir):
        self.dirname = dirname
        self.xy_data_dir = xy_data_dir
        self.cif_data_dir = cif_data_dir

    def read_in_data(self):
        taskname = "raw_text"
        raw_text = {}
        print("Reading in xy data...")
        for file in tqdm(os.listdir(self.xy_data_dir)):
            with open(os.path.join(self.xy_data_dir, file), 'r') as f:
                raw_text[file] = f.read()
        
        #checkpointing.save_checkpoint(self.dirname, taskname, raw_text, "txt")
        return raw_text

    def create_xrd_dataframe(self, raw_text):
        """
        Creates a dataframe from the raw text data.
        """

        xy_list = {'xrd': [], 'names': [], 'spec': [], 'lps': [], 'filename': []}
        for filename, txt in raw_text.items():
            xy_list['filename'].append(filename)

            xy_list['xrd'].append(extraction.extract_xy_from_string(txt))
            xy_list['names'].append(txt.split('##NAMES=')[1].split('\n')[0])
            if "##DIFFRACTION SAMPLE DESCRIPTION=" in txt:
                xy_list['spec'].append(txt.split('##DIFFRACTION SAMPLE DESCRIPTION=')[1].split('\n')[0])
            else:
                xy_list['spec'].append(None)
            if "CELL PARAMETERS" in txt:
                xy_list['lps'].append(strings.str_to_lattice(txt, dataset='RRUFF'))
            else:
                xy_list['lps'].append(None)

        xrd_df = pd.DataFrame(xy_list)
        #checkpointing.save_checkpoint(self.dirname, "xrd_df", xrd_df, "csv")
        
        return xrd_df
    
    def create_cif_dict(self, xrd_df):
        """
        Creates a dictionary with the names of the minerals as keys and the xy values as values.
        """
        
        print("Reading in cif data...")

        names = xrd_df['names']
        cif_dict = {}
        try:
            for filename in tqdm(os.listdir(self.cif_data_dir)):
                name = list(set([name for name in names if name in filename]))
            
                if name != []:
                    #take the name with the longest length
                    name = max(name, key=len)

                    try:
                        with open(os.path.join(self.cif_data_dir, filename), 'r') as f:
                            text = f.read()
                        if "data_global" in text:
                            cif_dict[filename] = {"name": name}

                    except Exception as e:
                        print(f"An error occurred while processing {filename}: {e}")
        
        except Exception as e:
            print(f"An error occurred during file listing: {e}")

        taskname = "cif_dict"
        checkpointing.save_checkpoint(self.dirname, taskname, cif_dict, "pt")

        return cif_dict
    
    def convert_cif_strings_to_structures(self, cif_dict):
        """
        Add pymatgen Structure objects to the xy_cif_dict.
        """

        print("Converting cif strings to structures...")

        for cif_string, value in tqdm(cif_dict.items()):
            file_path = os.path.join(self.cif_data_dir, cif_string)
            parser = CifParser((file_path))
            try: 
                structure = parser.get_structures()[0]
            except:
                structure = None
            value['structure'] = (structure)

        taskname = "cif_dict_with_pymatgen_structures"
        checkpointing.save_checkpoint(self.dirname, taskname, cif_dict, "pt")

        return cif_dict

    def extract_xy_from_string(data_string):
        """ 
        Extracts the x, y values from a string of data formatted as in the RRUFF dataset and returns them as a numpy array.
        """
        xy_values = []
        for line in data_string.splitlines():
            # Splitting the line at the comma and checking if it has exactly two elements
            parts = line.strip().split(',')
            if len(parts) == 2:
                try:
                    # Converting the parts to floats and appending to the list
                    x, y = float(parts[0].strip()), float(parts[1].strip())
                    xy_values.append([x, y])
                except ValueError:
                    # If conversion to float fails, move to the next line
                    continue
        return np.array(xy_values)
    
class checkpointing(): 
    def save_checkpoint(dirname, taskname, data, fmt = "pt"):
        if fmt == 'txt':
            filename = os.path.join(dirname, taskname + ".txt")
            with open(filename, 'w') as file:
                file.write(data)
        elif fmt == 'pt' or fmt == 'csv':
            filename = os.path.join(dirname, taskname + ".pt")
            torch.save(data, filename)
    
    def read_minerals_file_to_dict(file_path):
        minerals_dict = {}
        with open(file_path, 'r') as file:
            for line in file:
                # Splitting the line into key (mineral name) and value (number)
                parts = line.strip().split(":")
                if len(parts) == 2:
                    key = parts[0].strip().replace('"', '')  # Remove extra quotes and whitespace
                    value = parts[1].strip()
                    # Convert value to integer or None
                    minerals_dict[key] = int(value) if value.isdigit() else None
        return minerals_dict
    
class processing():
    """
    Suite of functions for processing data from a directory input of RRUFF data.
    """

    def __init__(self, dirname):
        self.dirname = dirname

    def remove_calculated_patterns(self, xrd_df):
        """
        remove structures that are calculated (not real experimental patterns)
        """

        keys_to_remove = []
        for key, row in xrd_df.iterrows():
            if row['spec'] is None: 
                keys_to_remove.append(key)   
            elif 'calculated' in row['spec'] or 'Calculated' in row['spec']:
                keys_to_remove.append(key)
            elif 'single' in row['spec'] or 'Single' in row['spec']:
                keys_to_remove.append(key)
        
        #remove the keys from xrd_df
        for key in keys_to_remove:
            xrd_df.drop(key, inplace=True)

        checkpointing.save_checkpoint(self.dirname, "no_calculated_patterns", xrd_df)

        return xrd_df

    def filter_structures(self, dictionary_with_structures, invert = False):
        """
        remove structures that are disordered or have more than 20 atoms from dictionary_with_structures
        """
        if not invert: 
            keys_to_remove = []
            for key, value in dictionary_with_structures.items():
                if value['structure'] is None: #if the structure is None
                    keys_to_remove.append(key)
                elif value['structure'].is_ordered == False: #if the structure is disordered
                    keys_to_remove.append(key)
                elif value['structure'].num_sites > 20: #if the structure has more than 20 atoms
                    keys_to_remove.append(key)
        else: 
            keys_to_remove = []
            for key, value in dictionary_with_structures.items():
                if value['structure'] is None: #if the structure is None
                    keys_to_remove.append(key)
                elif value['structure'].is_ordered == True: #if the structure is disordered
                    keys_to_remove.append(key)
                elif value['structure'].num_sites <= 20: #if the structure has more than 20 atoms
                    keys_to_remove.append(key)

        #remove the keys from dictionary_with_structures
        for key in keys_to_remove:
            dictionary_with_structures.pop(key)

        checkpointing.save_checkpoint(self.dirname, "filtered_structures", dictionary_with_structures)

        return dictionary_with_structures
    
    def remove_duplicates(self, dictionary_with_structures):
        """
        a function to identify structures in the dataset that are literally exactly identical by converting the structures to strings and comparing them.
        """

        #make a list of the structures as strings
        structures_as_strings = []
        for key, value in dictionary_with_structures.items():
            structures_as_strings.append(str(value['structure']))
            
        #find the indices of duplicates in the structures_as_strings list to remove. only keep the first instance of the duplicate
        indices = []
        for i in range(len(structures_as_strings)):
            for j in range(i+1, len(structures_as_strings)):
                if structures_as_strings[i] == structures_as_strings[j]:
                    indices.append(j)

        indices = list(set(indices))
        #remove the duplicates from the dictionary
        for i in sorted(indices, reverse=True):
            del dictionary_with_structures[list(dictionary_with_structures)[i]]

        checkpointing.save_checkpoint(self.dirname, "no_duplicates", dictionary_with_structures)

        return dictionary_with_structures
    
class selection():
    def plot_diffraction_pattern(index, dataframe, pv_dict_2, plot = True, eval_type = "cosine"):
        try: 
            #extract the experimental pattern 
            x = pv_dict_2[dataframe['material_id'].iloc[index]][0].numpy()[:,0][:8500]
            y = pv_dict_2[dataframe['material_id'].iloc[index]][0].numpy()[:,1][:8500]

            cif_string = dataframe['cif'].iloc[index]
            # Convert CIF string to Structure object
            structure = Structure.from_str(cif_string, fmt="cif")

            # Initialize the X-ray diffraction (XRD) calculator
            xrd_calculator = XRDCalculator(wavelength='CuKa')  # Using Copper K-alpha radiation by default

            # Calculate the diffraction pattern
            pattern = xrd_calculator.get_pattern(structure)
            sim_xrd = get_sim_xrd_from_pattern(pattern)

            if len(y) < 8500:
                x, y = adaption.convert_two_theta_array(x, y,  0.709, 1.54056)
                y = adaption.background_subtraction(x, y, 100)
                inter_fcn = adaption.interpolate(x, y) #default s is 0.0001
                x, y = adaption.change_domain(inter_fcn, x, [5, 90])
                
            #get the cosine similarity between the two 
            cosine_similarity_value = cosine_similarity(sim_xrd, y)

            #get the mse between the two
            mse_value = mse(sim_xrd, y)

            #plot the two patterns
            if plot:
                plot_xrds.plot(pattern, x, y)

        except Exception as e:
            print(e)
            cosine_similarity_value = 0
            mse_value = 1

        if eval_type == "cosine":
            return cosine_similarity_value
        elif eval_type == "mse":
            return -1 * mse_value

    def structure_and_xrd_plotter(counter, unique_material_ids, df_3, pv_dict_2, compare_metrics = False):
        chemical_name = unique_material_ids[counter]
        #find all indices of the chemical name in the df_3
        indices = []
        for i in range(len(df_3)):
            if df_3['material_id'].iloc[i] == chemical_name:
                indices.append(i)

        #plot all of the examples of the chemical name
        cosine_similarity_values = []
        mse_values = []
        structure_lattice = []
        provided_lattice = []
        provided_lattice.append(df_3['lattice'].iloc[0])

        for i in indices:
            cosine_similarity_values.append(selection.plot_diffraction_pattern(i, df_3, pv_dict_2, plot = False, eval_type = "cosine"))
            mse_values.append(selection.plot_diffraction_pattern(i, df_3, pv_dict_2, plot = False, eval_type = "mse"))

            structure_lattice.append(strings.str_to_lattice(df_3['cif'].iloc[i], dataset = 'MP'))
    
            # if RRUFF_amcsd_data.df['lps'] is not None: 
            #     length_mse = mse(np.array(RRUFF_amcsd_data.df['lps'].iloc[0].lengths) / np.max(np.array(RRUFF_amcsd_data.df['lps'].iloc[0].lengths)), np.array(my_lattice.lengths) / np.max(np.array(my_lattice.lengths)))
            #     angle_mse = mse(np.array(RRUFF_amcsd_data.df['lps'].iloc[0].angles) / np.max(np.array(RRUFF_amcsd_data.df['lps'].iloc[0].angles)), np.array(my_lattice.angles) / np.max(np.array(my_lattice.angles)))
            #     mse_val = length_mse + angle_mse
            #     lattice_mses.append(mse_val)
            # else:
            #     lattice_mses.append(100)

        cosine_sim_pick = indices[np.argmax(cosine_similarity_values)]
        negative_mse_pick = indices[np.argmax(mse_values)]

        if compare_metrics and not (cosine_sim_pick == negative_mse_pick):
            print("provided lattice")

            print("cosine sim pick")    
            print("cosine similarity: ", np.max(cosine_similarity_values))
            selection.plot_diffraction_pattern(cosine_sim_pick, df_3, pv_dict_2, plot = True)

            print("negative mse pick")
            print("negative mse: ", np.max(mse_values))
            selection.plot_diffraction_pattern(negative_mse_pick, df_3, pv_dict_2, plot = True)

        else:
            return cosine_sim_pick, negative_mse_pick

def is_monotonic_increasing(x):
    return np.all(x[:-1] <= x[1:])

def all_match(new_instance, cif_dict_with_structures):
    all_structures = list(cif_dict_with_structures.values())   

    all_names = list(set([all_structures[i]['name'] for i in range(len(all_structures))]))

    fulldf = pd.DataFrame()

    #reindex the xrd_df
    new_instance.xrd_df.reset_index(drop=True, inplace=True)

    for index in tqdm(range(0, len(all_names))):
        unique_material_id = all_names[index]

        #fix the indixes of new_instance.xrd_df
        new_instance.xrd_df = new_instance.xrd_df.reset_index(drop=True)

        #find all rows in new_instance.xrd_df that have "Friedelite" in the names column
        roi = new_instance.xrd_df[new_instance.xrd_df['names'].str.contains(unique_material_id)]
        
        if len(roi) > 0: 
        
            relevant_structures = [(i, all_structures[i]['structure']) for i in range(len(all_structures)) if all_structures[i]['name'] == unique_material_id]
            assert list(new_instance.cif_df['material_id']) == [all_structures[i]['name'] for i in range(len(all_structures))] #making sure the material ids are in the same order as the structures
            
            plot = False

            cosine_similarity_values = np.zeros((len(roi), len(relevant_structures)))
            mse_values = np.zeros((len(roi), len(relevant_structures)))

            # Initialize the X-ray diffraction (XRD) calculator
            xrd_calculator = XRDCalculator(wavelength='CuKa')  # Using Copper K-alpha radiation by default

            for i in range(len(roi)):
                for j in range(len(relevant_structures)):
                    
                    #extract the experimental pattern 
                    x = roi['xrd'].iloc[i][:,0][:8500]
                    y = roi['xrd'].iloc[i][:,1][:8500] / roi['xrd'].iloc[0][:,1].max()

                    #check to make sure x is monotonic
                    if is_monotonic_increasing(x):
                        structure = relevant_structures[j][1]

                        try: 
                            # if Element("Nh") in structure.species:
                            #     structure.replace_species({Element("Nh"): Element("N")})

                            # Calculate the diffraction pattern
                            pattern = xrd_calculator.get_pattern(structure)
                            sim_xrd = get_sim_xrd_from_pattern(pattern)

                            if len(y) < 8500:
                                    x, y = adaption.convert_two_theta_array(x, y,  0.709, 1.54056)
                                    y = adaption.background_subtraction(x, y, 100)
                                    inter_fcn = adaption.interpolate(x, y) #default s is 0.0001
                                    x, y = adaption.change_domain(inter_fcn, x, [5, 90])

                            #get the cosine similarity between the two 
                            cosine_similarity_value = cosine_similarity(sim_xrd, y)

                            #get the mse between the two
                            mse_value = mse(sim_xrd, y)

                            cosine_similarity_values[i, j] = cosine_similarity_value
                            mse_values[i, j] = mse_value
                        except Exception as e: 
                            print(e)
                            cosine_similarity_values[i, j] = 0
                            mse_values[i, j] = 1

                    else:
                        cosine_similarity_values[i, j] = 0
                        mse_values[i, j] = 1

            #find the indices of the maximum cosine similarity value as an i, j pair
            cosine_sim_pick = np.unravel_index(np.argmax(cosine_similarity_values), cosine_similarity_values.shape)

            #plot the xrd pattern of the cosine sim pick
            x = roi['xrd'].iloc[cosine_sim_pick[0]][:,0][:8500]
            y = roi['xrd'].iloc[cosine_sim_pick[0]][:,1][:8500] / roi['xrd'].iloc[0][:,1].max()

            structure = relevant_structures[cosine_sim_pick[1]][1]
            # Calculate the diffraction pattern
            pattern = xrd_calculator.get_pattern(structure)

            #plot_xrds.plot(pattern, x, y)     

            xrd_index = roi.index[cosine_sim_pick[0]]
            structures_index = relevant_structures[cosine_sim_pick[1]][0]

            xrd_row = new_instance.xrd_df.iloc[xrd_index]
            structure_row = new_instance.cif_df.iloc[structures_index]

            #combine the two rows into a single row
            combined_row = pd.concat([xrd_row, structure_row])

            #concatenate the two rows
            combined_row2 = pd.concat([xrd_row, structure_row], axis = 0)

            #add combined_row_2 to combined_row
            fulldf = pd.concat((fulldf, combined_row2), axis = 1)

    fulldf = fulldf.transpose()

    return fulldf

def viz_all_matches(fulldf):
    
    #plot the xrd and the pattern for every row in fulldf
    for index in range(len(fulldf)):
        print('mineral name: ', fulldf['material_id'].iloc[index])

        x = fulldf['xrd'].iloc[index][:,0][:8500]
        y = fulldf['xrd'].iloc[index][:,1][:8500] / fulldf['xrd'].iloc[0][:,1].max()

        if len(y) < 8500:
            x, y = adaption.convert_two_theta_array(x, y,  0.709, 1.54056)
            y = adaption.background_subtraction(x, y, 100)
            inter_fcn = adaption.interpolate(x, y)
            x, y = adaption.change_domain(inter_fcn, x, [5, 90])

        structure = fulldf['cif'].iloc[index]
        # Convert CIF string to Structure object
        structure = Structure.from_str(structure, fmt="cif")

        # if Element("Nh") in structure.species:
        #     structure.replace_species({Element("Nh"): Element("N")})

        # Initialize the X-ray diffraction (XRD) calculator
        xrd_calculator = XRDCalculator(wavelength='CuKa')  # Using Copper K-alpha radiation by default

        # Calculate the diffraction pattern
        pattern = xrd_calculator.get_pattern(structure)

        plot_xrds.plot(pattern, x, y / y.max())


def inspect_matches(new_instance, fulldf, unique_material_id):

    #fix the indixes of new_instance.xrd_df
    new_instance.xrd_df = new_instance.xrd_df.reset_index(drop=True)

    #find all rows in new_instance.xrd_df that have "Friedelite" in the names column
    roi = new_instance.xrd_df[new_instance.xrd_df['names'].str.contains(unique_material_id)]

    if len(roi) > 0: 
        
        #find all entries in new_instance.cif_dict that have "Friedelite" as the value associated with the key "name"
        all_structures = list(new_instance.cif_dict.values())

        relevant_structures = [(i, all_structures[i]['structure']) for i in range(len(all_structures)) if all_structures[i]['name'] == unique_material_id]
        assert list(new_instance.cif_df['material_id']) == [all_structures[i]['name'] for i in range(len(all_structures))] #making sure the material ids are in the same order as the structures
        
        plot = False

        cosine_similarity_values = np.zeros((len(roi), len(relevant_structures)))
        mse_values = np.zeros((len(roi), len(relevant_structures)))

        # Initialize the X-ray diffraction (XRD) calculator
        xrd_calculator = XRDCalculator(wavelength='CuKa')  # Using Copper K-alpha radiation by default

        for i in range(len(roi)):
            for j in range(len(relevant_structures)):

                #extract the experimental pattern 
                x = roi['xrd'].iloc[i][:,0][:8500]
                y = roi['xrd'].iloc[i][:,1][:8500] / roi['xrd'].iloc[0][:,1].max()

                structure = relevant_structures[j][1]

                if Element("Nh") in structure.species:
                    structure.replace_species({Element("Nh"): Element("N")})

                # Calculate the diffraction pattern
                pattern = xrd_calculator.get_pattern(structure)
                sim_xrd = get_sim_xrd_from_pattern(pattern)

                if len(y) < 8500:
                    x, y = adaption.convert_two_theta_array(x, y,  0.709, 1.54056)
                    y = adaption.background_subtraction(x, y, 100)
                    inter_fcn = adaption.interpolate(x, y) #default s is 0.0001
                    x, y = adaption.change_domain(inter_fcn, x, [5, 90])

                #get the cosine similarity between the two 
                cosine_similarity_value = cosine_similarity(sim_xrd, y)

                #get the mse between the two
                mse_value = mse(sim_xrd, y)

                cosine_similarity_values[i, j] = cosine_similarity_value
                mse_values[i, j] = mse_value

                print(f"i:{i}, j:{j}")

                print(roi.iloc[i])
                print("\n")
                print(structure)

                #plot
                plot_xrds.plot(pattern, x, y)



# THIS RENAMING BLOCK ONLY NEEDS TO BE RUN ONCE
# filenames = os.listdir("/home/gridsan/tmackey/cdvae/scripts/1-13-2024_RUFF_data/Combined/AMCSD")
# for file in filenames:
#     with open("/home/gridsan/tmackey/cdvae/scripts/1-13-2024_RUFF_data/Combined/AMCSD/" + file, 'r') as f:
#         try: 
#             text = f.read()
#             if "data_global" in text:
#                 #find what comes after _chemical_name_mineral
#                 mineral_name = text.split('_chemical_name_mineral')[1].split('\n')[0]

#                 #make a new filename that is just the mineral name .cif 
#                 new_filename = mineral_name + '.cif'
#                 how_many_exist = 0  
                
#                 #check if the file already exists
#                 if os.path.exists("/home/gridsan/tmackey/cdvae/scripts/1-13-2024_RUFF_data/Combined/AMCSD/" + new_filename):
#                     how_many_exist += 1
#                     new_filename = mineral_name + '_' + str(how_many_exist) + '.cif'
#                     os.rename("/home/gridsan/tmackey/cdvae/scripts/1-13-2024_RUFF_data/Combined/AMCSD/" + file, "/home/gridsan/tmackey/cdvae/scripts/1-13-2024_RUFF_data/Combined/AMCSD/" + new_filename)
#                 else:
#                     os.rename("/home/gridsan/tmackey/cdvae/scripts/1-13-2024_RUFF_data/Combined/AMCSD/" + file, "/home/gridsan/tmackey/cdvae/scripts/1-13-2024_RUFF_data/Combined/AMCSD/" + new_filename)

#             else:
#                 #remove the file if it doesn't have data_global
#                 os.remove("/home/gridsan/tmackey/cdvae/scripts/1-13-2024_RUFF_data/Combined/AMCSD/" + file) 
#         except:
#             print(file)      
        
    # def reformat_csv(self):
#     """
#     Reformat the dataframe to have the cif strings as strings instead of pymatgen structure objects.
#     """

#     def structure_to_cif_string(structure):
#         # Creating a CifWriter object
#         cif_writer = CifWriter(structure)

#         # Writing the structure to a string in CIF format
#         return cif_writer.__str__()

#     #apply the function to all of the keys in the dictionary
#     list_of_cifs = []
#     list_of_ids_2 = []
#     list_of_xrds_2 = []
#     list_of_specs_2 = []
#     list_of_lps_2 = []
#     list_of_structures = list(self.df['cif'])
#     list_of_names = list(self.df['material_id'])
#     list_of_specs = list(self.df['spec'])
#     list_of_lps = list(self.df['lps'])
    
#     for index, structure in enumerate(list_of_structures):
#         list_of_cifs.append(structure_to_cif_string(list_of_structures[index]))
#         list_of_ids_2.append(list_of_names[index])
#         list_of_xrds_2.append(self.list_of_xrds[index])
#         list_of_specs_2.append(list_of_specs[index])
#         list_of_lps_2.append(list_of_lps[index])

#     #make a new pandas dataframe with the cif string and the material id
#     self.df = pd.DataFrame({'cif': list_of_cifs, 'material_id': list_of_ids_2, 'spec': list_of_specs_2, 'lps': list_of_lps_2})

#     #for every material id remaining, find the corresponding code from the dictionary and use that as a key in a new dictionary
#     self.pv_dict= {}
#     for i in range(len(list_of_xrds_2)):
#         self.pv_dict[self.df['material_id'].iloc[i]] = list_of_xrds_2[i]

#     self.pv_dict = RRUFF_data.tensorize_xrds(self.pv_dict)

#     checkpointing.save_checkpoint(self.dirname, "reformatted_df", self.df, "csv")
#     checkpointing.save_checkpoint(self.dirname, "pv_dict", self.pv_dict, "pt")