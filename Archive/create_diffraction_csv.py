from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.diffraction.xrd import XRDCalculator

import pandas as pd

import sys

def generate_unique_id(structure):
        # Getting atomic species 
        atomic_species = [str(specie) for specie in structure.species]
        
        # Getting fractional coordinates
        frac_coords = structure.frac_coords
        #flatten the frac_coords list
        frac_coords = [coord for sublist in frac_coords for coord in sublist]
        #round all the frac_coords to 3 decimal places
        frac_coords = [round(coord, 7) for coord in frac_coords]
    
        # Getting lattice lengths and angles
        lattice_lengths = structure.lattice.abc
        lattice_angles = structure.lattice.angles
        
        #round all the lattice lengths and angles to 3 decimal places
        lattice_lengths = [round(length, 7) for length in lattice_lengths]
        lattice_angles = [round(angle, 7) for angle in lattice_angles]

        lattice_str = ''.join(map(str, lattice_lengths + lattice_angles))
        
        # Combining all the data
        # combined_data = ''.join(atomic_species) + frac_coords_str + lattice_str
        combined_data = ''.join(atomic_species) + lattice_str
            
        return combined_data

# Define a function to get XRD pattern
def calculate_xrd(structure):
    # Initialize the XRDCalculator with a wavelength of CuKa (1.54060 Å)
    xrd_calculator = XRDCalculator(wavelength='CuKa')

    pattern = xrd_calculator.get_pattern(structure)
    return pattern

def main(list_of_folder_directories, generate_patterns, change_TsachID):
    if generate_patterns == 'True':
        for folder_dir in list_of_folder_directories:

            df_train = pd.read_csv(folder_dir + 'train.csv')
            df_val = pd.read_csv(folder_dir + 'val.csv')
            df_test = pd.read_csv(folder_dir + 'test.csv')

            # Initialize the XRDCalculator
            xrd_calculator = XRDCalculator()

            df_train_structures = df_train['cif'].apply(lambda x: Structure.from_str(x, fmt='cif'))
            df_val_structures = df_val['cif'].apply(lambda x: Structure.from_str(x, fmt='cif'))
            df_test_structures = df_test['cif'].apply(lambda x: Structure.from_str(x, fmt='cif'))

            # Calculate XRD patterns for train, val, and test datasets
            df_train['xrd'] = df_train_structures.apply(calculate_xrd)
            df_val['xrd'] = df_val_structures.apply(calculate_xrd)
            df_test['xrd'] = df_test_structures.apply(calculate_xrd)

            #make a column with only the peak locations 
            df_train['xrd_peak_locations'] = df_train['xrd'].apply(lambda x: x.x.tolist())
            df_val['xrd_peak_locations'] = df_val['xrd'].apply(lambda x: x.x.tolist())
            df_test['xrd_peak_locations'] = df_test['xrd'].apply(lambda x: x.x.tolist())

            #make a column with only the peak intensities
            df_train['xrd_peak_intensities'] = df_train['xrd'].apply(lambda x: x.y.tolist())
            df_val['xrd_peak_intensities'] = df_val['xrd'].apply(lambda x: x.y.tolist())
            df_test['xrd_peak_intensities'] = df_test['xrd'].apply(lambda x: x.y.tolist())

            #make a column with only the atomic numbers (instead of the atomic species)
            df_train['atomic_numbers'] = df_train_structures.apply(lambda x: [Element(specie).Z for specie in x.species])
            df_val['atomic_numbers'] = df_val_structures.apply(lambda x: [Element(specie).Z for specie in x.species])
            df_test['atomic_numbers'] = df_test_structures.apply(lambda x: [Element(specie).Z for specie in x.species])

            #make a column called "TsachID" that creates a unique ID for each structure based on 
            # (a) atomic_species (b) frac_coordinates (c) lattice lengths and angles

            # Apply the unique ID generation function to the structures to create the "TsachID" column
            df_train['TsachID'] = df_train_structures.apply(generate_unique_id)
            df_val['TsachID'] = df_val_structures.apply(generate_unique_id)
            df_test['TsachID'] = df_test_structures.apply(generate_unique_id)

            #save the dataframes as csv files
            df_train.to_csv(folder_dir + 'train_xrd.csv')
            df_val.to_csv(folder_dir + 'val_xrd.csv')
            df_test.to_csv(folder_dir + 'test_xrd.csv')

            print('Finished creating XRD csv files for ' + folder_dir)
    
    if change_TsachID == 'True':
        for folder_name in list_of_folder_directories:
            #look for csv files that end with xrd.csv
            folders_that_end_with_xrd_csv = [folder_name + 'train_xrd.csv', folder_name + 'val_xrd.csv', folder_name + 'test_xrd.csv']
            for folder in folders_that_end_with_xrd_csv:
                df = pd.read_csv(folder)
                #rename the TsachID column to TsachID_old
                df = df.rename(columns={"TsachID": "TsachID_old"})
                #make a new TsachID column
                df['TsachID'] = df['cif'].apply(lambda x: generate_unique_id(Structure.from_str(x, fmt='cif')))

                df_structures = df['cif'].apply(lambda x: Structure.from_str(x, fmt='cif'))
  
                #make a column with only the atomic numbers (instead of the atomic species)
                df['atomic_numbers'] = df_structures.apply(lambda x: [Element(specie).Z for specie in x.species])

                duplicated_mask = df['TsachID'].duplicated(keep=False)
                if duplicated_mask.any():
                    print('ERROR: There are duplicates in the new TsachID column. Please check the code')
                    # print all occurrences of the duplicates
                    print(df[duplicated_mask])
                    sys.exit(1)

                
                #save the dataframe as a csv file
                df.to_csv(folder, index=False)
                print('Finished changing TsachIDs for ' + folder)

if __name__ == '__main__':
    #example command line call: python create_diffraction_csv.py True False /home/gridsan/tmackey/cdvae/data/perov_5/ /home/gridsan/tmackey/cdvae/data/mp_20/ /home/gridsan/tmackey/cdvae/data/carbon_24/

    #parse the fist argument which is "generate new diffraction patterns" and thats it
    generate_patterns = sys.argv[1]
    #parse the second argument which is "just change TsachID" and thats it
    change_TsachID = sys.argv[2]
    #parse the third argument which is the folder directory
    list_of_folder_directories = sys.argv[3:]

    # list_of_folder_directories = ['/home/gridsan/tmackey/cdvae/data/perov_5/', '/home/gridsan/tmackey/cdvae/data/mp_20/',
    #                           '/home/gridsan/tmackey/cdvae/data/carbon_24/']
    main(list_of_folder_directories, generate_patterns, change_TsachID)

