from pymatgen.core.lattice import Lattice
import re

class strings(): 
    def str_to_lattice(params_str, dataset):
        """
        Args:
            params_str (str): String containing the cell parameters. 
                Ex: ##CELL PARAMETERS=a: 13.463(3) b: 13.463(3) c: 5.914(2) alpha: 90 beta: 90 gamma: 120 volume: 928.3(4) crystal system: hexagonal
            dataset (str): Name of the dataset from which the cell parameters were extracted
        """

        if dataset == 'RRUFF':
            # Parse the cell parameters
            params = {
                'a:': None,
                'b:': None,
                'c:': None,
                'alpha:': None,
                'beta:': None,
                'gamma:': None
            }

            #look for the number immediately after the key and store it in the dictionary
            for key in params:
                match = re.search(key + r'\s*([\d.]+)', params_str)
                if match:
                    params[key] = float(match.group(1))

            # Create a monoclinic lattice using the extracted parameters
            # In pymatgen, for a monoclinic lattice, alpha and gamma are 90 by default if not specified
            lattice = Lattice.from_parameters(a=params['a:'], b=params['b:'], c=params['c:'],
                                            alpha=params.get('alpha:', 90), beta=params['beta:'], gamma=params.get('gamma:', 90))

            return lattice
        
        elif dataset == 'MP':
            # Parse the cell parameters
            params = {
                '_cell_length_a   ': None,
                '_cell_length_b   ': None,
                '_cell_length_c   ': None,
                '_cell_angle_alpha   ': None,
                '_cell_angle_beta   ': None,
                '_cell_angle_gamma   ': None
            }

            #look for the number immediately after the key and store it in the dictionary
            for key in params:
                match = re.search(key + r'\s*([\d.]+)', params_str)
                if match:
                    params[key] = float(match.group(1))

            # Create a monoclinic lattice using the extracted parameters
            # In pymatgen, for a monoclinic lattice, alpha and gamma are 90 by default if not specified
            keys = list(params.keys())
            lattice = Lattice.from_parameters(a=params[keys[0]], b=params[keys[1]], c=params[keys[2]],
                                            alpha=params[keys[3]], beta=params[keys[4]], gamma=params[keys[5]])

            return lattice

