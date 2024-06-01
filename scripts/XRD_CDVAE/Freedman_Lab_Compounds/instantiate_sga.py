# wrapper_functions.py
from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer
from pymatgen.core.structure import Structure

def instantiate_spacegroup_analyzer(structure, tolerance=0.01, angle_tolerance=5):
    """
    Wrapper function for instantiating SpacegroupAnalyzer with additional parameters.
    
    Args:
        structure (Structure): A pymatgen Structure object.
        symprec (float): The symmetry precision.
        angle_tolerance (float): The angle tolerance.
        
    Returns:
        An instance of SpacegroupAnalyzer.
    """
    sga = SpacegroupAnalyzer(structure, symprec=tolerance, angle_tolerance=angle_tolerance)
    return sga