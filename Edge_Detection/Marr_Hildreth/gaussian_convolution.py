from PIL import Image
import numpy as np
import math
from scipy import signal

def gaussian_function(array, sigma):
    return np.exp(-((pow(array,2)))/(2*(pow(sigma,2))))  # Gaussian approximation

# Returns a 1D Gaussian filter for a given value of sigma
def gauss1d(sigma):
    filter_length = 6 * sigma
    if(filter_length % 2 == 0):
        filter_length = filter_length + 1
    
    minimum = (-1) * (math.ceil(filter_length / 2) - 1)
    maximum = (-1) * minimum
    
    filter_array = np.arange(minimum, maximum + 1)
    filter_array = filter_array.astype(float)
    vfunc = np.vectorize(gaussian_function)
    gaussian_filter = vfunc(filter_array, sigma)
    normalization_factor = np.sum(gaussian_filter)                              
    gaussian_filter = gaussian_filter/normalization_factor
    return gaussian_filter

def gauss2d(sigma):
    one_dg = gauss1d(sigma)
    one_dg = one_dg[np.newaxis]
    transp = np.transpose(one_dg)
    result = signal.convolve2d(one_dg, transp)
    return result

def gaussconvolve2d(array, sigma):
    filter_array = gauss2d(sigma)
    result = signal.convolve2d(array, filter_array, 'same')
    return result
