"""Utility functions to protect main process against segfaults"""
from __future__ import absolute_import, division, print_function, unicode_literals
from multiprocessing import Pool
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

pool = Pool(processes=1)


def segfault_protect(function, *args, **kwargs):
    """Used to run a function in a separate process to catch RuntimeErrors such
    as segfaults.
    """
    result = pool.apply_async(function, (args))
    res = result.get()
    return res

def start_spacegroupanalyzer(self, structure, tolerance, angle_tolerance):
        return SpacegroupAnalyzer(structure, symprec=tolerance, angle_tolerance=angle_tolerance)