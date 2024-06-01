import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from tqdm import tqdm
from evaluate_utils.eval_utils import * 
from scipy.interpolate import UnivariateSpline
from numpy.polynomial import chebyshev
from scipy import optimize
from math import pi

def caglioti_fwhm(theta, U, V, W):
    """
    Calculate the FWHM using the Caglioti formula.
    theta: float, the angle in degrees
    U, V, W: Caglioti parameters
    """
    rad_theta = np.radians(theta / 2)  # Convert theta to radians
    return (U * np.tan(rad_theta)**2 + V * np.tan(rad_theta) + W)**0.5

def pseudo_voigt(x, center, amplitude, U, V, W, eta, noise_sd=0.0):
    """
    Pseudo-Voigt function using Caglioti FWHM.
    x: array-like, the independent variable
    center: float, the center of the peak
    amplitude: float, the height of the peak
    U, V, W: Caglioti parameters
    eta: float, the fraction of the Lorentzian component (0 <= eta <= 1)
    """
    fwhm = caglioti_fwhm(center, U, V, W)
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma for Gaussian
    # Generate random noise from a normal distribution
    noise = np.random.normal(0, noise_sd)

    noisy_percentage = (100 + noise_sd) / 100 
    #print("noisy_percentage is ", noisy_percentage)

    #multiply the amplitude by the noisy percentage 
    amplitude = amplitude * noisy_percentage
    
    lorentzian = amplitude * (fwhm**2 / ((x - center)**2 + fwhm**2))
    gaussian = amplitude * np.exp(-(x - center)**2 / (2 * sigma**2))
    return eta * lorentzian + (1 - eta) * gaussian

def superimposed_pseudo_voigt(x, xy_merge, U, V, W, eta, noise_sd=0.0):
    """
    Superimpose multiple pseudo-Voigt functions using Caglioti FWHM.
    x: array-like, the independent variable
    xy_merge: nx2 array, first column is peak locations, second column is intensities
    U, V, W: Caglioti parameters
    eta: float, the fraction of the Lorentzian component (0 <= eta <= 1)
    """
    total = np.zeros_like(x)
    for row in xy_merge:
        center, amplitude = row
        total += pseudo_voigt(x, center, amplitude, U, V, W, eta, noise_sd)
    total = total / max(total)
    return total

# Function to simulate XRD for each row
def simulate_pv_xrd_for_row(xy_merge, U, V, W, noise=0.0):
    """
    Simulate a pseudo-Voigt XRD pattern for a given set of peaks.

    Args:
    xy_merge: array-like, the 2D array of peak locations and intensities
    U, V, W: float, the Caglioti parameters
    noise: float, the standard deviation of the noise

    Returns:
    sim_xrd: array-like, the simulated XRD pattern
    """

    x = np.arange(5, 90, 0.010)
    eta = 0  # Fraction of Lorentzian component (common for all peaks)

    sim_xrd = superimposed_pseudo_voigt(x, xy_merge, U, V, W, eta, noise_sd=noise)

    return sim_xrd

def get_sim_xrd_from_crystal_list(cryslist, index):
    """
    Get the simulated XRD pattern for a single crystal in the list.
    cryslist: list of Crystal objects
    index: int, the index of the crystal in the list
    """
    crys = Crystal(cryslist[index])
    pattern =  xrd_calculator.get_pattern(crys.structure)
    sim_xrd = get_sim_xrd_from_pattern(pattern)
    return sim_xrd

def get_sim_xrd_from_pattern(pattern):
    """
    Get the simulated XRD pattern for pymatgen generated pattern object
    """

    xy_merge = np.column_stack((pattern.x, pattern.y))

    U, V, W = 0.1, 0.1, 0.1 
    sim_xrd = simulate_pv_xrd_for_row(xy_merge, U, V, W, noise=0.0)
    return sim_xrd

def cosine_similarity(x, y):
    """
    Compute the cosine similarity between two vectors.
    x, y: array-like, the two vectors
    """
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def mse(x, y):
    """
    Compute the mean squared error between two vectors.
    x, y: array-like, the two vectors
    """
    return np.mean((x - y)**2)

def get_cosine_similarities_from_crystal_lists(best_crystals, gt_crystals):
    """
    Get the cosine similarities between the simulated XRD patterns for two lists of crystals.
    cryslist1: list of Crystal objects
    cryslist2: list of Crystal objects
    """
    consine_similarity_values = []
    for i in tqdm(range(len(best_crystals))): 
        pred_pv_xrd = get_sim_xrd_from_crystal_list(best_crystals, i)
        gt_pv_xrd = get_sim_xrd_from_crystal_list(gt_crystals, i)
        consine_similarity_values.append(cosine_similarity(pred_pv_xrd, gt_pv_xrd))

    return consine_similarity_values


class adaption(): 
      
    q_min = 0.5
    q_max = 0.2
    wavelength = 1.5406

    def convert_two_theta_array(two_theta, intensities, lambda_1, lambda_2):
        """
        Convert a 2θ array from one wavelength to another.
        two_theta: array-like, the 2θ array
        intensities: array-like, the intensity array
        lambda_1: float, the initial wavelength
        lambda_2: float, the new wavelength
        """

        # Convert 2θ array to θ array for the initial wavelength
        theta_1 = np.radians(two_theta / 2)
        
        # Calculate d-spacing using the initial wavelength for each θ
        d = lambda_1 / (2 * np.sin(theta_1))
        
        # Calculate the new θ array using the new wavelength
        theta_2 = np.arcsin(lambda_2 / (2 * d))
        
        # Convert θ array back to 2θ array
        two_theta_2 = 2 * np.degrees(theta_2)

        #delete any nan values
        indices = ~np.isnan(two_theta_2)
        two_theta_2 = two_theta_2[indices]
        intensities = intensities[indices]
        
        return two_theta_2, intensities

    @DeprecationWarning
    def background_subtraction_rolling_median(two_theta, intensity, window_size=100):
        """
        Background subtraction of the XRD data using a rolling median filter.

        Args:
        pvxrd: DataFrame, the XRD data
        window_size: int, the size of the rolling window
        s: factor for the spline https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.html

        Returns:
        background_subtracted: array-like, the background-subtracted intensity data
        """

        # Convert data to DataFrame for rolling average calculation
        df = pd.DataFrame({'2theta': two_theta, 'intensity': intensity})
        
        # Compute rolling average for background
        background = df['intensity'].rolling(window=window_size, min_periods=1, center=True).median()
        
        # Subtract background
        background_subtracted = df['intensity'] - background
        
        # Ensure no negative values after background subtraction
        background_subtracted[background_subtracted < 0] = 0
 
        return background_subtracted
    
    def exclude_outliers(x, y, fit_y, std_mult = 3):
        y_bs = y - fit_y
        mean = np.mean(y_bs)
        std = np.std(y_bs)
        outlier_mask = (y_bs < mean + std * std_mult) & (y_bs > mean - std * std_mult)
        x_data = x[outlier_mask]
        y_data = y[outlier_mask]
        if len(y_data) < 10:
            #print("Too many outliers are being excluded")
            x_data = x
            y_data = y
        return x_data, y_data

    def chebyshev_fit(x, *coeffs):
        # Use the first 3 Chebyshev polynomials of the first kind
        return sum(c * chebyshev.chebval(x, [0] * i + [1]) for i, c in enumerate(coeffs))

    def background_subtraction(y_data, cheb_order = 5, std_mult = 3, tolerance = 1e-8):

        start_2theta = np.arcsin((adaption.q_min * adaption.wavelength) / (4 * pi)) * 360 / pi
        stop_2theta = np.arcsin((adaption.q_max * adaption.wavelength) / (4 * pi)) * 360 / pi
        step_size = (stop_2theta - start_2theta) / len(y_data)

        x_init = np.arange(len(y_data)) * step_size + start_2theta

        x_data = x_init

        y_data = y_data / np.max(y_data)

        background = y_data
    
        fit_y = np.zeros(len(background))
        initial_guess = [0] * cheb_order
        counter = 0
       
        while np.abs(np.sum(fit_y - background)/len(fit_y)) > tolerance and counter < 20:
            popt, _ = optimize.curve_fit(adaption.chebyshev_fit, x_data, background, p0=initial_guess)
            fit_y = adaption.chebyshev_fit(x_data, *popt)
            x_data, background = adaption.exclude_outliers(x_data, background, fit_y, std_mult=std_mult)
            fit_y = adaption.chebyshev_fit(x_data, *popt)
            counter += 1
    
        if counter == 20:
            print("This model may perform poorly on data with a complicated background")
            # plt.plot(x_init, y_data, label='Data')
            # plt.show()
        
        popt, _ = optimize.curve_fit(adaption.chebyshev_fit, x_data, background, p0=initial_guess)

        # Generate fitted y values
        fit_y = adaption.chebyshev_fit(x_init, *popt)

        # Subtract fitted y values from data
        y_data = y_data - fit_y

        # Ensure no negative values after background subtraction
        y_data[y_data < 0] = 0

        return y_data
    
    def interpolate(two_theta, background_subtracted, s=0.0001):
        """
        Interpolate the background-subtracted XRD data using a spline.
        
        Args:
        df: DataFrame, the XRD data
        background_subtracted: array-like, the background-subtracted intensity data
        s: factor for the spline https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.html

        Returns:
        interp_func: function, the interpolated function
        """

         # Interpolate using scipy spline
        interp_func = UnivariateSpline(two_theta, background_subtracted, s=s)

        return interp_func

    def change_domain(inter_fcn, original_two_theta, new_domain = [5, 90]): 
        """
        Change the domain of the XRD data to be between 5 and 90 degrees. Uses the interpolation functions to do this.

        Args:
        inter_fcn: function, the interpolated function from UniVariteSpline
        original_two_theta: array-like, the original 2theta data (defines the range we can reasonably interpolate over)
        new_domain: list, the new domain for the XRD data [min, max]

        Returns:
        new_df: DataFrame, the new XRD data with the new domain
        
        """
        x_new = np.arange(new_domain[0], new_domain[1], 0.01)
        y_new = inter_fcn(x_new)

        #zero out values outside the range of the original data
        y_new[x_new < min(original_two_theta)] = 0
        y_new[x_new > max(original_two_theta)] = 0

        #normalize AFTER zeroing out values
        y_new = y_new / max(y_new)

        return x_new, y_new

        