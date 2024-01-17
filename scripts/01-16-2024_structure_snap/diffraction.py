from math import pi
import torch
import numpy as np
import torch.nn.functional as Fun
import math
import gc
import matplotlib.pyplot as plt

class diffraction_pattern():
	def __init__(self, q_min = 0.5, q_max = 0.2, wavelength = 1.5406):
		self.q_min = q_min
		self.q_max = q_max
		self.wavelength = wavelength

	atom_form_factor_constants = [
		[[0.202, 30.868], [0.244, 8.544], [0.082, 1.273], [0, 0]],
		[[0.091, 18.183], [0.181, 6.212], [0.11, 1.803], [0.036, 0.284]],
		[[1.611, 107.638], [1.246, 30.48], [0.326, 4.533], [0.099, 0.495]],
		[[1.25, 60.804], [1.334, 18.591], [0.36, 3.653], [0.106, 0.416]],
		[[0.945, 46.444], [1.312, 14.178], [0.419, 3.223], [0.116, 0.377]],
		[[0.731, 36.995], [1.195, 11.297], [0.456, 2.814], [0.125, 0.346]],
		[[0.572, 28.847], [1.043, 9.054], [0.465, 2.421], [0.131, 0.317]],
		[[0.455, 23.78], [0.917, 7.622], [0.472, 2.144], [0.138, 0.296]],
		[[0.387, 20.239], [0.811, 6.609], [0.475, 1.931], [0.146, 0.279]],
		[[0.303, 17.64], [0.72, 5.86], [0.475, 1.762], [0.153, 0.266]],
		[[2.241, 108.004], [1.333, 24.505], [0.907, 3.391], [0.286, 0.435]],
		[[2.268, 73.67], [1.803, 20.175], [0.839, 3.013], [0.289, 0.405]],
		[[2.276, 72.322], [2.428, 19.773], [0.858, 3.08], [0.317, 0.408]],
		[[2.129, 57.775], [2.533, 16.476], [0.835, 2.88], [0.322, 0.386]],
		[[1.888, 44.876], [2.469, 13.538], [0.805, 2.642], [0.32, 0.361]],
		[[1.659, 36.65], [2.386, 11.488], [0.79, 2.469], [0.321, 0.34]],
		[[1.452, 30.935], [2.292, 9.98], [0.787, 2.234], [0.322, 0.323]],
		[[1.274, 26.682], [2.19, 8.813], [0.793, 2.219], [0.326, 0.307]],
		[[3.951, 137.075], [2.545, 22.402], [1.98, 4.532], [0.482, 0.434]],
		[[4.47, 99.523], [2.971, 22.696], [1.97, 4.195], [0.482, 0.417]],
		[[3.966, 88.96], [2.917, 20.606], [1.925, 3.856], [0.48, 0.399]],
		[[3.565, 81.982], [2.818, 19.049], [1.893, 3.59], [0.483, 0.386]],
		[[3.245, 76.379], [2.698, 17.726], [1.86, 3.363], [0.486, 0.374]],
		[[2.307, 78.405], [2.334, 15.785], [1.823, 3.157], [0.49, 0.364]],
		[[2.747, 67.786], [2.456, 15.674], [1.792, 3.0], [0.498, 0.357]],
		[[2.544, 64.424], [2.343, 14.88], [1.759, 2.854], [0.506, 0.35]],
		[[2.367, 61.431], [2.236, 14.18], [1.724, 2.725], [0.515, 0.344]],
		[[2.21, 58.727], [2.134, 13.553], [1.689, 2.609], [0.524, 0.339]],
		[[1.579, 62.94], [1.82, 12.453], [1.658, 2.504], [0.532, 0.333]],
		[[1.942, 54.162], [1.95, 12.518], [1.619, 2.416], [0.543, 0.33]],
		[[2.321, 65.602], [2.486, 15.458], [1.688, 2.581], [0.599, 0.351]],
		[[2.447, 55.893], [2.702, 14.393], [1.616, 2.446], [0.601, 0.342]],
		[[2.399, 45.718], [2.79, 12.817], [1.529, 2.28], [0.594, 0.328]],
		[[2.298, 38.83], [2.854, 11.536], [1.456, 2.146], [0.59, 0.316]],
		[[2.166, 33.899], [2.904, 10.497], [1.395, 2.041], [0.589, 0.307]],
		[[2.034, 29.999], [2.927, 9.598], [1.342, 1.952], [0.589, 0.299]],
		[[4.776, 140.782], [3.859, 18.991], [2.234, 3.701], [0.868, 0.419]],
		[[5.848, 104.972], [4.003, 19.367], [2.342, 3.737], [0.88, 0.414]],
		[[4.129, 27.548], [3.012, 5.088], [1.179, 0.591], [0, 0]],
		[[4.105, 28.492], [3.144, 5.277], [1.229, 0.601], [0, 0]],
		[[4.237, 27.415], [3.105, 5.074], [1.234, 0.593], [0, 0]],
		[[3.12, 72.464], [3.906, 14.642], [2.361, 3.237], [0.85, 0.366]],
		[[4.318, 28.246], [3.27, 5.148], [1.287, 0.59], [0, 0]],
		[[4.358, 27.881], [3.298, 5.179], [1.323, 0.594], [0, 0]],
		[[4.431, 27.911], [3.343, 5.153], [1.345, 0.592], [0, 0]],
		[[4.436, 28.67], [3.454, 5.269], [1.383, 0.595], [0, 0]],
		[[2.036, 61.497], [3.272, 11.824], [2.511, 2.846], [0.837, 0.327]],
		[[2.574, 55.675], [3.259, 11.838], [2.547, 2.784], [0.838, 0.322]],
		[[3.153, 66.649], [3.557, 14.449], [2.818, 2.976], [0.884, 0.335]],
		[[3.45, 59.104], [3.735, 14.179], [2.118, 2.855], [0.877, 0.327]],
		[[3.564, 50.487], [3.844, 13.316], [2.687, 2.691], [0.864, 0.316]],
		[[4.785, 27.999], [3.688, 5.083], [1.5, 0.581], [0, 0]],
		[[3.473, 39.441], [4.06, 11.816], [2.522, 2.415], [0.84, 0.298]],
		[[3.366, 35.509], [4.147, 11.117], [2.443, 2.294], [0.829, 0.289]],
		[[6.062, 155.837], [5.986, 19.695], [3.303, 3.335], [1.096, 0.379]],
		[[7.821, 117.657], [6.004, 18.778], [3.28, 3.263], [1.103, 0.376]],
		[[4.94, 28.716], [3.968, 5.245], [1.663, 0.594], [0, 0]],
		[[5.007, 28.283], [3.98, 5.183], [1.678, 0.589], [0, 0]],
		[[5.085, 28.588], [4.043, 5.143], [1.684, 0.581], [0, 0]],
		[[5.151, 28.304], [4.075, 5.073], [1.683, 0.571], [0, 0]],
		[[5.201, 28.079], [4.094, 5.081], [1.719, 0.576], [0, 0]],
		[[5.255, 28.016], [4.113, 5.037], [1.743, 0.577], [0, 0]],
		[[5.267, 28.016], [4.113, 5.037], [1.743, 0.577], [0, 0]],
		[[5.225, 29.158], [4.314, 5.259], [1.827, 0.586], [0, 0]],
		[[5.272, 29.046], [4.347, 5.226], [1.844, 0.585], [0, 0]],
		[[5.332, 28.888], [4.37, 5.198], [1.863, 0.581], [0, 0]],
		[[5.376, 28.773], [4.403, 5.174], [1.884, 0.582], [0, 0]],
		[[5.436, 28.655], [4.437, 5.117], [1.891, 0.577], [0, 0]],
		[[5.441, 29.149], [4.51, 5.264], [1.956, 0.59], [0, 0]],
		[[5.529, 28.927], [4.533, 5.144], [1.945, 0.578], [0, 0]],
		[[5.553, 28.907], [4.58, 5.16], [1.969, 0.577], [0, 0]],
		[[5.588, 29.001], [4.619, 5.164], [1.997, 0.579], [0, 0]],
		[[5.659, 28.807], [4.63, 5.114], [2.014, 0.578], [0, 0]],
		[[5.709, 28.782], [4.677, 5.084], [2.019, 0.572], [0, 0]],
		[[5.695, 28.968], [4.74, 5.156], [2.064, 0.575], [0, 0]],
		[[5.75, 28.933], [4.773, 5.139], [2.079, 0.573], [0, 0]],
		[[5.754, 29.159], [4.851, 5.152], [2.096, 0.57], [0, 0]],
		[[5.803, 29.016], [4.87, 5.15], [2.127, 0.572], [0, 0]],
		[[2.388, 42.866], [4.226, 9.743], [2.689, 2.264], [1.255, 0.307]],
		[[2.682, 42.822], [4.241, 9.856], [2.755, 2.295], [1.27, 0.307]],
		[[5.932, 29.086], [4.972, 5.126], [2.195, 0.572], [0, 0]],
		[[3.51, 52.914], [4.552, 11.884], [3.154, 2.571], [1.359, 0.321]],
		[[3.841, 50.261], [4.679, 11.999], [3.192, 2.56], [1.363, 0.318]],
		[[6.07, 28.075], [4.997, 4.999], [2.232, 0.563], [0, 0]],
		[[6.133, 28.047], [5.031, 4.957], [2.239, 0.558], [0, 0]],
		[[4.078, 38.406], [4.978, 11.02], [3.096, 2.355], [1.326, 0.299]],
		[[6.201, 28.2], [5.121, 4.954], [2.275, 0.556], [0, 0]],
		[[6.215, 28.382], [5.17, 5.002], [2.316, 0.562], [0, 0]],
		[[6.278, 28.323], [5.195, 4.949], [2.321, 0.557], [0, 0]],
		[[6.264, 28.651], [5.263, 5.03], [2.367, 0.563], [0, 0]],
		[[6.306, 28.688], [5.303, 5.026], [2.386, 0.561], [0, 0]],
		[[6.767, 85.951], [6.729, 15.642], [4.014, 2.936], [1.561, 0.335]],
		[[6.323, 29.142], [5.414, 5.096], [2.453, 0.568], [0, 0]],
		[[6.415, 28.836], [5.419, 5.022], [2.449, 0.561], [0, 0]],
		[[6.462, 28.396], [5.469, 4.97], [2.471, 0.554], [0, 0]],
		[[6.46, 28.396], [5.469, 4.97], [2.471, 0.554], [0, 0]],
		[[6.502, 28.375], [5.478, 4.975], [2.51, 0.561], [0, 0]],
		[[6.548, 28.461], [5.526, 4.965], [2.52, 0.557], [0, 0]],
		[[6.58, 28.543], [5.572, 4.958], [2.538, 0.555], [0, 0]],
		[[6.61, 28.621], [5.615, 4.954], [2.553, 0.553], [0, 0]],
		[[6.638, 28.695], [5.655, 4.952], [2.566, 0.552], [0, 0]],
		[[6.665, 28.765], [5.692, 4.952], [2.577, 0.551], [0, 0]],
		[[6.69, 28.832], [5.727, 4.954], [2.587, 0.55], [0, 0]],
		[[6.713, 28.896], [5.759, 4.958], [2.596, 0.549], [0, 0]],
		[[6.735, 28.957], [5.789, 4.963], [2.604, 0.548], [0, 0]],
		[[6.756, 29.015], [5.817, 4.969], [2.611, 0.547], [0, 0]],
		[[6.776, 29.071], [5.843, 4.976], [2.618, 0.546], [0, 0]],
		[[6.795, 29.124], [5.867, 4.984], [2.624, 0.545], [0, 0]],
		[[6.813, 29.175], [5.89, 4.992], [2.63, 0.544], [0, 0]],
		[[6.831, 29.224], [5.912, 5.001], [2.635, 0.543], [0, 0]],
		[[6.848, 29.271], [5.932, 5.01], [2.64, 0.542], [0, 0]],
		[[6.864, 29.316], [5.951, 5.019], [2.645, 0.541], [0, 0]],
		[[6.88, 29.36], [5.969, 5.028], [2.649, 0.541], [0, 0]],
		[[6.895, 29.402], [5.986, 5.038], [2.653, 0.54], [0, 0]],
		[[6.909, 29.443], [6.002, 5.047], [2.657, 0.539], [0, 0]],
		[[6.923, 29.483], [6.017, 5.057], [2.661, 0.539], [0, 0]],
		[[6.936, 29.522], [6.031, 5.066], [2.664, 0.538], [0, 0]],
		[[6.949, 29.56], [6.044, 5.076], [2.667, 0.538], [0, 0]],
	]

	"""	def batch_split_diffraction_calc(self, batch, batch_size):
		pattern_list = []

		# Add extra padding to account for rounding problems
		max_hkl_pts_length = self.evaluate_max_hkl_points(batch) + 6000

		for i in range(math.ceil(batch['angles'].size()[0]/batch_size)):
			if i < math.ceil(batch['angles'].size()[0]/batch_size) - 1:
				indices = [i*batch_size,(i+1)*batch_size]
			else:
				indices = [i*batch_size, batch['angles'].size()[0]]
			pattern_list.append(self.diffraction_calc(batch, max_hkl_pts_length, indices = indices))
		pattern = torch.cat(pattern_list, dim=1)
		return pattern"""

	def diffraction_calc(self, batch, max_hkl_pts_length = None, indices = None):

		# default q_max should be 4 * pi / wavelength
		# Calculate the cell matrix

		cell_matrix = self.get_cell_matrix(batch, indices = indices)
		# Cell matrix is a [batch_size, 3, 3] tensor

		# Get all the hkl points that satisfy the chosen q_max and put them in a list
		# Also put the r of the lattice point in a different list
		hkl_pts, recip_lengths, max_hkl_pts_length = self.collect_recip_latt_points(cell_matrix, max_hkl_pts_length)
		# hkl_pts is a [batch_size, max_hkl_pts_length, 3] tensor
		# recip_lengths is a [batch_size, max_hkl_pts_length] tensor


		# Free up memory
		del cell_matrix

		# Collect all the information on the atoms
		fcoords, occus, zs, coeffs = self.get_fcoords_occus_zs_coeffs(batch, indices)
		# fcoords is a [sum_atoms, 3] tensor
		# occus is a [batch_size, sum_atoms, max_atom_type] tensor
		# zs is a [max_atom_type] tensor
		# coeffs is a [max_atom_type, 4, 2] tensor


		# Calculate the atom positions times each set of hkl indices
		g_dot_r = torch.matmul(hkl_pts, fcoords.T)
		# g_dot_r is a [batch_size, max_hkl_pts_length, sum_atoms] tensor

		# Free up memory
		del hkl_pts, fcoords

		# Calculate the atomic scattering factors at each r (or angle)
		fs = self.calc_atomic_scattering_factor(zs, recip_lengths, coeffs)
		# fs is a [batch_size, max_atom_type, max_hkl_pts_length] tensor

		# Free up memory
		del zs, coeffs

		# Multiply the atomic scattering factors by the occupation of each site
		w_fs = torch.matmul(occus, fs)
		# w_fs is a [batch_size, sum_atoms, max_hkl_pts_length] tensor

		# Free up memory
		del occus, fs

		# Calculate the complex phase and intensity of the scattering for each hkl
		f_hkl = torch.sum(torch.transpose(w_fs, 1, 2) * torch.exp(2j * pi * g_dot_r), axis = 2)
		# f_hkl is a [batch_size, max_hkl_pts_length] tensor

		# Free up memory
		del g_dot_r, w_fs

		# Calculate the real instensity for each hkl
		i_hkl = (f_hkl * torch.conj(f_hkl)).real
		# i_hkl is a [batch_size, max_hkl_pts_length] tensor

		# Free up memory
		del f_hkl

		# Calculate the lorentz factor based on the recip lengths
		lorentz_factor = self.calc_lorentz_factor(recip_lengths)
		# lorentz_factor is a [batch_size, max_hkl_pts_length] tensor

		# Calculate two theta values from recip lengths and turn into degrees from rad
		# two_thetas = 2 * torch.asin(wavelength * recip_lengths / 2) * 180 / pi
		two_thetas = (360 / pi) * torch.asin(self.wavelength * recip_lengths / 2)

		# two_thetas is a [batch_size, max_hkl_pts_length] tensor

		#Calculate intensities from i_hkl and correction factor (lorentz_factor)
		intensities = i_hkl * lorentz_factor

		# Free up memory
		del recip_lengths, i_hkl, lorentz_factor

		# Wrap the two thetas and intensities together
		pattern = torch.stack((two_thetas, intensities))
		
		return pattern

	"""
	def evaluate_max_hkl_points(self, batch):
		cell_matrix = self.get_cell_matrix(batch)

		hkl_pts, recip_lengths, max_hkl_pts_length = self.collect_recip_latt_points(cell_matrix)

		# Free up memory
		del cell_matrix, recip_lengths, hkl_pts

		return max_hkl_pts_length"""


	def get_cell_matrix(self, batch, indices = None):
		try:
			angles = batch['angles'][indices[0]:indices[1]]
			lengths = batch['lengths'][indices[0]:indices[1]]
			angles_rad = angles * pi / 180

		except:
			angles = batch['angles']
			lengths = batch['lengths']
			angles_rad = angles * pi / 180

		a_vector = torch.mul(lengths[:,0].unsqueeze(1), torch.tensor([1, 0, 0], dtype=torch.float32).cuda().unsqueeze(0))
		b_vector = torch.mul(lengths[:,1].unsqueeze(1), (torch.cos(angles_rad[:,2]).unsqueeze(1) * torch.tensor([1,0,0]).cuda() + torch.sin(angles_rad[:,2]).unsqueeze(1) * torch.tensor([0,1,0]).cuda()))
		c_vector = torch.mul(lengths[:,2].unsqueeze(1), (torch.cos(angles_rad[:,1]).unsqueeze(1) * torch.tensor([1,0,0]).cuda() + ((torch.cos(angles_rad[:,0]) - torch.cos(angles_rad[:,1]) * torch.cos(angles_rad[:,2])) /  torch.sin(angles_rad[:,2])).unsqueeze(1) * torch.tensor([0,1,0]).cuda() + torch.sqrt(1 - torch.square(torch.cos(angles_rad[:,1])) - torch.square((torch.cos(angles_rad[:,0]) - torch.cos(angles_rad[:,1]) * torch.cos(angles_rad[:,2])) / torch.sin(angles_rad[:,2]))).unsqueeze(1) * torch.tensor([0,0,1]).cuda()))
		cell_matrix = torch.stack((a_vector,b_vector, c_vector), dim=1)
		# Free up memory
		del a_vector, b_vector, c_vector, angles, lengths, angles_rad

		return cell_matrix


	def collect_recip_latt_points(self, cell_matrix, max_hkl_pts_length = None):
		

		# Obtained from Bragg condition. Note that reciprocal lattice
		# vector length is 1 / d_hkl.
		recip_latt = torch.transpose(torch.linalg.inv(cell_matrix), 1, 2)

		max_r = self.q_max / (2 * pi)

		max_h = torch.round(max_r / torch.linalg.norm(recip_latt[:,0], dim=1))
		max_k = torch.round(max_r / torch.linalg.norm(recip_latt[:,1], dim=1))
		max_l = torch.round(max_r / torch.linalg.norm(recip_latt[:,2], dim=1))

		# Get all the hkl points that satisfy the chosen q_max and put them in a list
		hkl_pts, max_hkl_pts_length = self.get_points_in_sphere(max_h, max_k, max_l, recip_latt, max_hkl_pts_length)
		
		recip_pts = torch.matmul(hkl_pts, recip_latt)
		recip_lengths = torch.linalg.norm(recip_pts, dim = 2)

		# Free up memory
		del recip_latt, max_r, max_h, max_k, max_l, recip_pts
		
		return hkl_pts, recip_lengths, max_hkl_pts_length


	def get_points_in_sphere(self, max_h, max_k, max_l, recip_latt, max_hkl_pts_length = None):
		# This calculates for peaks that are within rectangular prism that circumscribes sphere
		# Make a matrix the size of the reciprocal lattice
		indices = torch.stack((max_h, max_k, max_l), dim=1)
		recip_coords = []
		for i, hkl in enumerate(indices):
			# Create a matrix of ones representing the reciprocal lattice
			hkl_ones_matrix = torch.ones((int(hkl[0].item()*2) + 1, int(hkl[1].item()*2) + 1, int(hkl[2].item())*2 + 1)).cuda()

			# Get the indices of every point in that lattice and flatten it into a [n, 3] list
			hkl_pts = torch.nonzero(hkl_ones_matrix, as_tuple=False).cuda() - torch.tensor([hkl[0].item(), hkl[1].item(), hkl[2].item()]).cuda()

			# Calculate the recip lengths to avoid calculations for unphysical lengths
			recip_pts = torch.matmul(hkl_pts, recip_latt[i])
			recip_lengths = torch.linalg.norm(recip_pts, dim = 1)

			# Filter out unphysical lengths
			hkl_pts_in_sphere = hkl_pts[(self.wavelength * recip_lengths / 2) < 1]

			# Remove the incident beam from the list of hkl points (point at the center)
			center = torch.tensor([0, 0, 0]).cuda()
			hkl_pts_in_sphere = hkl_pts_in_sphere[torch.any(hkl_pts_in_sphere != center, dim=1)]
			
			recip_coords.append(hkl_pts_in_sphere)
		
		if max_hkl_pts_length is None:
			max_hkl_pts_length = 0
			for hkl_pts in recip_coords:
				if hkl_pts.size()[0] > max_hkl_pts_length:
					max_hkl_pts_length = hkl_pts.size()[0]
		
		#hkl_pts = torch.zeros((max_h.size()[0], max_hkl_pts_length, 3), dtype=torch.int64).cuda()

		padded_recip_coords = []
		for hkl_pts in recip_coords:
			hkl_pts = torch.cat((hkl_pts, torch.ones((max_hkl_pts_length - hkl_pts.size()[0], 3), dtype=torch.int64).cuda()/0), dim=0)
			hkl_pts = hkl_pts.unsqueeze(0)
			padded_recip_coords.append(hkl_pts)
		
		hkl_pts = torch.cat(padded_recip_coords, dim=0)

		# Free up memory
		del indices, recip_coords, center, hkl_ones_matrix, hkl_pts_in_sphere, padded_recip_coords

		return hkl_pts, max_hkl_pts_length


	def get_fcoords_occus_zs_coeffs(self, batch, indices):
		if indices is None:
			indices = [0, batch['num_atoms'].size()[0]]
		num_atoms = batch['num_atoms'][indices[0]:indices[1]]
		num_atoms_begin = torch.sum(batch['num_atoms'][0:indices[0]])
		num_atoms_end = torch.sum(batch['num_atoms'][0:indices[1]])

		fcoords = batch['frac_coords'][num_atoms_begin:num_atoms_end]
		atom_types = batch['atom_types'][num_atoms_begin:num_atoms_end]
		occus = Fun.one_hot(atom_types - 1).float()

		# Zero out coeffs in front of atoms which don't exist in the crystal
		occus = torch.ones((num_atoms.size()[0], 1, 1)).cuda() * occus.unsqueeze(0)
		for i in range(num_atoms.size()[0]):
			occus[i, 0:torch.sum(num_atoms[0:i]), :] = 0
			occus[i, torch.sum(num_atoms[0:i+1]):-1, :] = 0

		# Get the maximum atom type in the structure
		max_atom_type = torch.max(atom_types)
		zs = range(1, max_atom_type + 1)
		zs = torch.tensor(zs).cuda()

		# Collect the coefficients for each element in the structure into a torch tensor from the atom_form_factor_constants dictionary
		coeffs = torch.tensor(self.atom_form_factor_constants, dtype=torch.float32).cuda()
		coeffs = coeffs[zs - 1]

		# Free up memory
		del atom_types, max_atom_type

		return fcoords, occus, zs, coeffs


	def calc_atomic_scattering_factor(self, zs, recip_lengths, coeffs):
		#Add additional dimension to the r^2 list for each hkl
		squared_recip_lengths = torch.square(recip_lengths/2).unsqueeze(1)
		# squared_recip_lengths is a [batch_size, 1, max_hkl_pts_length] tensor

		# Calculate the exponentiated component of the fitted atomic scattering factor
		extinct_exp = torch.exp(-coeffs[:, :, 1].unsqueeze(2) * squared_recip_lengths.unsqueeze(1))
		# extinct_exp is a [batch_size, max_atom_type, 4, max_hkl_pts_length] tensor
		
		# Calculate the whole of the fitted atomic scattering factor function
		fitted_factor = torch.sum(coeffs[:, :, 0].unsqueeze(2) * extinct_exp, axis = 2)
		# fitted_factor is a [batch_size, max_atom_type, max_hkl_pts_length] tensor

		# Calculate the full angle dependent component of the atomic scattering factor
		angle_fs = 41.78214 * squared_recip_lengths * fitted_factor
		# angle_fs is a [batch_size, max_atom_type, max_hkl_pts_length] tensor

		# Calculate the atomic scattering factor for each hkl
		fs = zs.unsqueeze(0).unsqueeze(2) - angle_fs
		# fs is a [batch_size, max_atom_type, max_hkl_pts_length] tensor

		# Free up memory
		del squared_recip_lengths, extinct_exp, fitted_factor, angle_fs
		
		return fs

	def calc_lorentz_factor(self, recip_lengths):
		# Calculate a theta list from r's
		theta = torch.asin(self.wavelength * recip_lengths / 2)

		# Calculate how the intensity of peaks are expected to tail off due to geometric constraints of experiments
		lorentz_factor = (1 + torch.square(torch.cos(2 * theta))) / (torch.square(torch.sin(theta)) * torch.cos(theta))

		"""Another way to calculate it without trig functions is
		x = wavelength * recip_lengths / 2
		#lorentz_factor = (4 * torch.square(torch.square(x)) - 4 * torch.square(x) + 2) / (torch.square(x) * torch.sqrt(1 - torch.square(x)))
		"""

		# Free up memory
		del theta

		return lorentz_factor
	
	def batch_split_bin_pattern_theta(self, pattern, batch_size, U = 0, V = 0, W = 0.1, num_steps = 256, fraction_gaussian = 0.5):
		binned_pattern_list = []
		for i in range(math.ceil(pattern.size()[1]/batch_size)):
			print("Batch: ", i)
			if i < math.ceil(pattern.size()[1]/batch_size) - 1:
				indices = [i*batch_size, (i+1)*batch_size]
			else:
				indices = [i*batch_size, pattern.size()[1]]
			if i == 0:
				binned_pattern = self.bin_pattern_theta(pattern, indices = indices, U = U, V = V, W = W, num_steps = num_steps, fraction_gaussian = fraction_gaussian)
			else:
				binned_pattern = torch.cat([binned_pattern, self.bin_pattern_theta(pattern, indices = indices, U = U, V = V, W = W, num_steps = num_steps, fraction_gaussian = fraction_gaussian)])
				print(binned_pattern.size())
			gc.collect()
			torch.cuda.empty_cache()
		return binned_pattern


	def bin_pattern_theta(self, pattern, indices = None, U = 0, V = 0, W = 0.1, num_steps = 256, fraction_gaussian = 1):

		def caglioti_fwhm(pattern, U, V, W):
			"""
			Calculate the FWHM using the Caglioti formula.
			theta: float, the angle in degrees
			U, V, W: Caglioti parameters
			"""
			rad_theta = torch.deg2rad(pattern / 2)  # Convert theta to radians
			caglioti_fwhm = torch.sqrt(U * torch.square(torch.tan(rad_theta)) + V * torch.tan(rad_theta) + W)

			# Free up memory
			del rad_theta

			return caglioti_fwhm
		
		if indices is not None:
			pattern = pattern[:, indices[0]:indices[1], :]
		
		# This function is written to work in degrees, it can be used for q instead, but fwhm should change, and the step size should be calculated differently

		# Calculate the theta range
		two_theta_min = np.arcsin((self.q_min * self.wavelength) / (4 * pi)) * 360 / pi
		two_theta_max = np.arcsin((self.q_max * self.wavelength) / (4 * pi)) * 360 / pi

		# Calculate the width of each bin
		step_size = (two_theta_max - two_theta_min) / num_steps

		# Make a tensor for the domain of the pattern
		pattern_domain = torch.arange(num_steps).cuda() * step_size + two_theta_min

		# Calculate fwhm based on angle
		fwhm = caglioti_fwhm(pattern[0,:,:], U, V, W)
		#fwhm = torch.tensor([1]).cuda()

		# Calculate prefactors for the pseudo-Voigt peak
		#ag = (2 / fwhm) * np.sqrt(np.log(2) / pi)
		ag = 0.9394372787 / fwhm

		#bg = (4 * np.log(2)) / (fwhm**2)
		bg = 2.77258872224 / (fwhm**2)


		# Setup a domain for each peak in diffraction_pattern
		peaks = torch.ones(pattern.size()[1], 1, pattern.size()[2]).cuda() * pattern_domain.unsqueeze(1)
		# peaks is a [batch_size, num_steps, num_peaks] tensor

		# Calculate the gaussian component of the peak
		gaussian_peak = fraction_gaussian * ag.unsqueeze(1) * torch.exp(-bg.unsqueeze(1) * torch.square(peaks - pattern[0,:,:].unsqueeze(1)))
		# gaussian_peak is a [batch_size, num_steps, num_peaks] tensor

		# Free up memory
		#del pattern_domain, ag, bg
		del ag, bg

		# Calculate the lorentzian component of the peak
		lorentz_peak = (1 - fraction_gaussian) * fwhm.unsqueeze(1) / (6.28318530718 * (torch.square(peaks - pattern[0,:,:].unsqueeze(1)) + torch.square(fwhm.unsqueeze(1)/2)))

		# Free up memory
		del fwhm, peaks

		# Calculate the combined peak
		combined_peak = pattern[1,:,:].unsqueeze(1) * (gaussian_peak + lorentz_peak)

		# Free up memory
		del gaussian_peak, lorentz_peak

		# Sum over all peaks to get the binned pattern
		binned_pattern = torch.nansum(combined_peak, axis = 2)

		# Free up memory
		del combined_peak

		return binned_pattern, pattern_domain

	def calculate_loss(self, pattern, initial_structure, U = 0, V = 0, W = 0.1, q_min = None, q_max = None):
		if q_min is not None:
			self.q_min = q_min
		if q_max is not None:
			self.q_max = q_max
		num_steps = pattern.size()[0]
		binned_initial_struct_pattern, pattern_domain = self.bin_pattern_theta(self.diffraction_calc(initial_structure), U = U, V = V, W = W, num_steps = num_steps)
		loss = -1*Fun.cosine_similarity(pattern, binned_initial_struct_pattern[0], dim = 0)
		return loss, binned_initial_struct_pattern, pattern_domain


	def snap_structure(self, pattern, initial_structure, learning_rate = 0.01, iterations = 100, U = 0, V = 0, W = 1, sharpen_rate = 0, min_U = 0, min_V = 0, min_W = 0.2, plot_progress = False, plot_freq = 10):
		snap_lengths = True
		snap_angles = True
		snap_coords = True
		# Gradient descent function using PyTorch autograd
		for i in range(iterations):
			if U > min_U:
				U = U / (sharpen_rate + 1)
			if V > min_V:
				V = V / (sharpen_rate + 1)
			if W > 0.1:
				W = W / (sharpen_rate + 1)
			loss, simulated_pattern, simulated_domain = self.calculate_loss(pattern, initial_structure, U = U, V = V, W = W)
			loss.backward()
			with torch.no_grad():
				if snap_lengths:
					initial_structure['lengths'] -= learning_rate * initial_structure['lengths'].grad
					initial_structure['lengths'].grad.zero_() # Clear the gradients for the next 

				if snap_angles:
					initial_structure['angles'] -= learning_rate * initial_structure['angles'].grad
					initial_structure['angles'].grad.zero_() # Clear the gradients for the next 

				if snap_coords:
					initial_structure['frac_coords'] -= learning_rate * initial_structure['frac_coords'].grad
					initial_structure['frac_coords'].grad.zero_() # Clear the gradients for the next 

			if plot_progress and i % plot_freq == 0:
				self.plot_progress(simulated_domain, simulated_pattern, pattern)
				print(f"Step {i+1}: Lengths - {initial_structure['lengths']}, Angles - {initial_structure['angles']}, Frac Coords - {initial_structure['frac_coords']}, Loss - {loss.item()}")
		
		snapped_structure = initial_structure

		return snapped_structure

	def plot_progress(self, simulated_domain, simulated_pattern, pattern):
		plt.plot(simulated_domain.cpu().detach().numpy(), simulated_pattern[0].cpu().detach().numpy(), label = 'Simulated Pattern')
		plt.plot(simulated_domain.cpu().detach().numpy(), pattern.cpu(), label = 'Experimental Pattern')
		plt.show()
		plt.close()

	def pymatgen_pattern(self, pattern):
		two_thetas = [0]
		pymat_pattern = torch.tensor([[0,0]]).cuda()
		for i in range(pattern.size()[1]):
			if pattern[1, i] > 0.001:
				#print(pattern[:,i])
				ind = torch.where(torch.abs(torch.tensor(two_thetas).cuda() - pattern[0, i]) < 0.1)
				if ind[0].size()[0] > 0:
					pymat_pattern[ind[0][0]][1] = pymat_pattern[ind[0][0]][1] + pattern[1][i]
				else:
					pymat_pattern = torch.cat((pymat_pattern, pattern.T[i].unsqueeze(0)))
					two_thetas.append(pattern[0][i])
		pymat_pattern[1:].sort(axis = 0)
		return pymat_pattern