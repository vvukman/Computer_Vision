from PIL import Image
import numpy as np
import math
from scipy import signal

def xderivative(image_array):  
    image_array_copy = image_array.copy()
    xfilter = [1, -1]
    for row in range(0, len(image_array)):
        for col in range(0, len(image_array[0])):
            if(col >= math.floor(len(xfilter)/2) and col <= len(image_array[0]) - math.floor(len(xfilter)/2) - 1):
                delta = (image_array[row, col] * xfilter[0]) + (image_array[row, col + 1] * xfilter[1])
                image_array_copy[row, col] = delta
            else:
                image_array_copy[row, col] = 0
    return image_array_copy

def yderivative(image_array):  
    image_array_copy = image_array.copy()
    yfilter = [1, -1]
    for row in range(0, len(image_array)):
        for col in range(0, len(image_array[0])):
            if(row >= math.floor(len(yfilter)/2) and row <= len(image_array) - math.floor(len(yfilter)/2) - 1):
                delta = (image_array[row, col] * yfilter[0]) + (image_array[row + 1, col] * yfilter[1])
                image_array_copy[row, col] = delta
            else:
                image_array_copy[row, col] = 0
    return image_array_copy
