"""This module implements an XRD pattern calculator."""

from __future__ import annotations
import torch
import json
import os
from math import asin, cos, degrees, pi, radians, sin
from typing import TYPE_CHECKING

import numpy as np

# Preliminary steps
# XRD wavelengths in angstroms
with open(os.path.join(os.path.dirname(__file__), "atomic_scattering_params.json")) as f:
	ATOMIC_SCATTERING_PARAMS = json.load(f)

atom_names = {
	1: "H",
	2: "He",
	3: "Li",
	4: "Be",
	5: "B",
	6: "C",
	7: "N",
	8: "O",
	9: "F",
	10: "Ne",
	11: "Na",
	12: "Mg",
	13: "Al",
	14: "Si",
	15: "P",
	16: "S",
	17: "Cl",
	18: "Ar",
	19: "K",
	20: "Ca",
	21: "Sc",
	22: "Ti",
	23: "V",
	24: "Cr",
	25: "Mn",
	26: "Fe",
	27: "Co",
	28: "Ni",
	29: "Cu",
	30: "Zn",
	31: "Ga",
	32: "Ge",
	33: "As",
	34: "Se",
	35: "Br",
	36: "Kr",
	37: "Rb",
	38: "Sr",
	39: "Y",
	40: "Zr",
	41: "Nb",
	42: "Mo",
	43: "Tc",
	44: "Ru",
	45: "Rh",
	46: "Pd",
	47: "Ag",
	48: "Cd",
	49: "In",
	50: "Sn",
	51: "Sb",
	52: "Te",
	53: "I",
	54: "Xe",
	55: "Cs",
	56: "Ba",
	57: "La",
	58: "Ce",
	59: "Pr",
	60: "Nd",
	61: "Pm",
	62: "Sm",
	63: "Eu",
	64: "Gd",
	65: "Tb",
	66: "Dy",
	67: "Ho",
	68: "Er",
	69: "Tm",
	70: "Yb",
	71: "Lu",
	72: "Hf",
	73: "Ta",
	74: "W",
	75: "Re",
	76: "Os",
	77: "Ir",
	78: "Pt",
	79: "Au",
	80: "Hg",
	81: "Tl",
	82: "Pb",
	83: "Bi",
	84: "Po",
	85: "At",
	86: "Rn",
	87: "Fr",
	88: "Ra",
	89: "Ac",
	90: "Th",
	91: "Pa",
	92: "U",
	93: "Np",
	94: "Pu",
	95: "Am",
	96: "Cm",
	97: "Bk",
	98: "Cf"
}


def get_cell_matrix(structure):
	angles = torch.tensor(structure[0], requires_grad=True)
	lengths = torch.tensor(structure[1], requires_grad=True)
	angles_rad = angles * pi / 180

	a_vector = torch.tensor([1, 0, 0]) * lengths[0]
	b_vector = (torch.tensor([1,0,0]) * torch.cos(angles_rad[2]) + torch.tensor([0,1,0]) * torch.sin(angles_rad[2])) * lengths[1]
	c_vector = (torch.tensor([1,0,0]) * torch.cos(angles_rad[1]) + torch.tensor([0,1,0]) * ((torch.cos(angles_rad[0]) - torch.cos(angles_rad[1]) * torch.cos(angles_rad[2])) /  torch.sin(angles_rad[2])) + torch.tensor([0,0,1]) * torch.sqrt(1 - torch.square(torch.cos(angles_rad[1])) - torch.square((torch.cos(angles_rad[0]) - torch.cos(angles_rad[1]) * torch.cos(angles_rad[2])) / torch.sin(angles_rad[2]))) )* lengths[2]
	cell_matrix = torch.stack((a_vector,b_vector, c_vector))
	return cell_matrix


def collect_recip_latt_points(cell_matrix, q_max):
	# Obtained from Bragg condition. Note that reciprocal lattice
	# vector length is 1 / d_hkl.
	recip_latt = torch.linalg.inv(cell_matrix).T

	max_r = q_max / (2 * pi)

	max_h = round(max_r / float(torch.linalg.norm(recip_latt[0])))
	max_k = round(max_r / float(torch.linalg.norm(recip_latt[1])))
	max_l = round(max_r / float(torch.linalg.norm(recip_latt[2])))

	hkl_pts = get_points_in_sphere(recip_latt, max_h, max_k, max_l)
	recip_pts = torch.mm(hkl_pts, recip_latt)
	recip_lengths = torch.linalg.vector_norm(recip_pts, dim = 1)
	return hkl_pts, recip_lengths


def get_points_in_sphere(recip_latt, max_h, max_k, max_l):
	# TODO: Only do calculation for peaks that are within radius r of sphere
	# Right now, this calculates for peaks that are within rectangular prism that circumscribes sphere
	hkl_ones_matrix = torch.ones((max_h*2 + 1, max_k * 2 + 1, max_l * 2 + 1))
	hkl_pts = torch.argwhere(hkl_ones_matrix) - torch.Tensor([max_h, 0, 0]) - torch.Tensor([0, max_k, 0]) - torch.Tensor([0, 0, max_l])
	return hkl_pts


def get_fcoords_occus_zs_coeffs(structure):
	fcoords = torch.tensor(structure[2], requires_grad = True)
	occus = torch.tensor(structure[3], requires_grad = True)
	zs = torch.tensor(structure[4])
	coeffs = torch.ones((2, 4, 2))
	#coeffs = torch.ones((#of unique elements, 4, 2))
	return fcoords, occus, zs, coeffs


def calc_atomic_scattering_factor(zs, recip_lengths, coeffs):
	#Add additional dimension to the r^2 list for each hkl
	squared_recip_lengths = torch.square(recip_lengths).unsqueeze(0)

	# Calculate the exponentiated component of the fitted atomic scattering factor
	extinct_exp = torch.exp(-coeffs[:, :, 1].unsqueeze(2) * squared_recip_lengths)

	# Calculate the whole of the fitted atomic scattering factor function
	fitted_factor = torch.sum(coeffs[:, :, 0].unsqueeze(2) * extinct_exp, axis = 1)

	# Calculate the full angle dependent component of the atomic scattering factor
	angle_fs = 41.78214 * squared_recip_lengths * fitted_factor

	# Calculate the atomic scattering factor for each hkl
	fs = zs.unsqueeze(1) - angle_fs

	return fs

def calc_lorentz_factor(recip_lengths, wavelength):
	# Calculate a theta list from r's
	theta = torch.asin(wavelength * recip_lengths / 2)

	# Calculate how the intensity of peaks are expected to tail off due to geometric constraints of experiments
	lorentz_factor = (1 + torch.square(torch.cos(2 * theta))) / (torch.square(torch.sin(theta)) * torch.cos(theta))

	"""Another way to calculate it without trig functions is
	x = wavelength * recip_lengths / 2
	lorentz_factor = -(4 * torch.square(torch.square(x)) + 4 * torch.square(x) - 2) / (torch.square(x) * torch.sqrt(1 - torch.square(x)))
	"""

	return lorentz_factor



def diffraction_calc(structure, q_max, wavelength = 1.54184):
	#default q_max should be 4 * pi / wavelength
	# Calculate the cell matrix
	cell_matrix = get_cell_matrix(structure)

	# Get all the hkl points that satisfy the chosen q_max and put them in a list
	# Also put the r of the lattice point in a different list
	hkl_pts, recip_lengths = collect_recip_latt_points(cell_matrix, q_max)

	# Collect all the information on the atoms
	fcoords, occus, zs, coeffs = get_fcoords_occus_zs_coeffs(structure)

	# Calculate the atom positions times each set of hkl indices
	g_dot_r = torch.mm(hkl_pts, fcoords.T)

	# Calculate the atomic scattering factors at each r (or angle)
	fs = calc_atomic_scattering_factor(zs, recip_lengths, coeffs)

	# Multiply the atomic scattering factors by the occupation of each site
	w_fs = torch.mm(occus, fs)

	# Calculate the complex phase and intensity of the scattering for each hkl
	f_hkl = torch.sum(w_fs.T * torch.exp(2j * pi * g_dot_r), axis = 1)

	# Calculate the real instensity for each hkl
	i_hkl = (f_hkl * torch.conj(f_hkl)).real

	# Calculate the lorentz factor based on the recip lengths
	lorentz_factor = calc_lorentz_factor(recip_lengths, wavelength)

	# Calculate two theta values from recip lengths
	two_thetas = 2 * torch.asin(wavelength * recip_lengths / 2)

	#Calculate intensities from i_hkl and correction factor (lorentz_factor)
	intensities = i_hkl * lorentz_factor

	pattern = torch.stack((two_thetas, intensities))

	return pattern



def diffraction_loss(structure_gen, structure_ref, q_max = 8):
	ref_pattern = diffraction_calc(structure_ref, q_max)
	gen_pattern = diffraction_calc(structure_gen, q_max)


angles_ref = [90.0,90.0,90.0]
lengths_ref = [4.0,4.0,4.0]
atom_positions_ref = [[0,0,0],[.5,.5,.5],[0.75,0.75,0.75]]
atom_types_ref = [[0.25,0.75],[0.25,0.75],[0.5,0.5]]
zs_ref = [1,2]

angles_gen = [90.0,90.0,90.0]
lengths_gen = [4.0,4.0,4.0]
atom_positions_gen = [[0,0,0],[0.5,0.5,0.49]]
atom_types_gen = [[0.25,0.75],[0.25,0.75]]
zs_gen = [1,2]

structure_ref = [angles_ref, lengths_ref, atom_positions_ref, atom_types_ref, zs_ref]
structure_gen = [angles_gen, lengths_gen, atom_positions_gen, atom_types_gen, zs_gen]

diffraction_loss(structure_gen, structure_ref)