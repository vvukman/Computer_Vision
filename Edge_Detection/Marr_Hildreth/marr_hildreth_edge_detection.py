from PIL import Image
import numpy as np
import math
from scipy import signal
import gaussian_convolution as gc
import mh_image_derivatives as id 

def marr_hildreth_edge_detector(image, sigma):
    image_array = np.asarray(image)
    # Smooth the image by Gaussian filter
    smoothed_image_array = gc.gaussconvolve2d(image_array, sigma)
    # Apply Laplacian to smoothed image 
    laplacian_of_img = laplacian(smoothed_image_array)
    # Find zero crossings
    result = find_zero_crossings(laplacian_of_img)
    return result

def find_zero_crossings(laplacian_of_img):
    result = np.zeros((laplacian_of_img.shape[0], laplacian_of_img.shape[1]), dtype=np.int)
    # Array indicating if values are positive or negative
    image_array_signs = np.sign(laplacian_of_img)
    
    # Difference along xaxis
    xdiff = np.diff(image_array_signs, axis=1)
    xzero_crossings = np.where(xdiff)
    # Output of where gives two arrays...combine the result to obtain [x,y] coordinate pairs
    xzero_crossings = np.dstack((xzero_crossings[0], xzero_crossings[1]))[0]
        
    #difference along yaxis
    ydiff = np.diff(image_array_signs, axis=0)
    yzero_crossings = np.where(ydiff)
    # Output of where gives two arrays...combine the result to obtain [x,y] coordinate pairs
    yzero_crossings = np.dstack((yzero_crossings[0], yzero_crossings[1]))[0]

    xzero_crossings_rows = xzero_crossings.view([('', xzero_crossings.dtype)] * xzero_crossings.shape[1])
    yzero_crossings_rows = yzero_crossings.view([('', yzero_crossings.dtype)] * yzero_crossings.shape[1])
    # Obtain the tuples of xzero_crossings which are not found in yzero_crossings
    diff = np.setdiff1d(xzero_crossings_rows, yzero_crossings_rows).view(xzero_crossings_rows.dtype).reshape(-1, xzero_crossings_rows.shape[1])

    # The format of diff cannot be used in append due to different "shape" of yzero_crossings and diff.
    diff_formatted = []
    for index in range(0, len(diff)):
        diff_formatted.append(diff[index][0]) 
    diff_a, diff_b = zip(*diff_formatted)
    difference_result = np.dstack((diff_a, diff_b))[0]

    # Append the zero crossings inside yzero_crossings with the remaining x,y coordinates
    zero_crossings = np.append(yzero_crossings, difference_result, axis=0)
    for tuple in zero_crossings:
        result[tuple[0], tuple[1]] = 120
    return result

def laplacian(image_array):
    #take second derivative in the x direction
    fxx = id.xderivative(id.xderivative(image_array))
    #take second derivative in the y direction
    fyy = id.yderivative(id.yderivative(image_array))
    return fxx + fyy


image = Image.open('/.../parthenon.jpg')
image = image.convert('L')
result = marr_hildreth_edge_detector(image, 0.8)
result = result.astype('uint8')
image = Image.fromarray(result)
image.save('/.../marr_hildreth_edge_detection_parthenon.jpg','JPEG')