from PIL import Image
import numpy as np
import math
from scipy import signal
import gaussian_convolution as gc
import canny_image_derivatives as cid 

n_map = {44: [0,1],89: [-1,1],134: [-1,0],179: [-1,-1],224: [0,-1],269: [1,-1],314: [1,0],359: [1,1]}


def obtain_neighbour(gradient_array, row, col):
    try:
        return [gradient_array[row][col][0], row, col]
    except IndexError:
        return None


def is_maximum(row, col, pixel, gradient_array):
    na = [0,0]
    nb = [0,0]
    gradient_magnitude = pixel[0]
    gradient_direction = pixel[1]
    if(gradient_direction < 0):
        gradient_direction = 360 + np.rad2deg(gradient_direction)
    
    if(gradient_direction <= 44):
        if(col < len(gradient_array[0])-1):
            na = n_map[44] 
        if(col > 0):           
            nb = np.negative(n_map[44])
    elif(gradient_direction <= 89):
        if(row > 0 and col < len(gradient_array[0])-1):
            na = n_map[89] 
        if(row < len(gradient_array)-1 and col > 0):           
            nb = np.negative(n_map[89])
    elif(gradient_direction <= 134):
        if(row > 0):
            na = n_map[134] 
        if(row < len(gradient_array)-1):           
            nb = np.negative(n_map[134])
    elif(gradient_direction <= 179):
        if(row > 0 and col > 0):
            na = n_map[179] 
        if(row < len(gradient_array)-1 and col < len(gradient_array[0])-1):           
            nb = np.negative(n_map[179])
    elif(gradient_direction <= 224):
        if(col > 0):
            na = n_map[224] 
        if(col < len(gradient_array[0])-1):           
            nb = np.negative(n_map[224])
    elif(gradient_direction <= 269):
        if(row < len(gradient_array)-1 and col > 0):
            na = n_map[269] 
        if(row > 0 and col < len(gradient_array[0])-1):           
            nb = np.negative(n_map[269])
    elif(gradient_direction <= 314):
        if(row < len(gradient_array)-1):
            na = n_map[314] 
        if(row > 0):           
            nb = np.negative(n_map[314])
    elif(gradient_direction <= 359):
        if(row < len(gradient_array)-1 and col < len(gradient_array[0])-1):
            na = n_map[359] 
        if(row > 0 and col > 0):           
            nb = np.negative(n_map[359])
    
    na_magnitude = gradient_array[row + na[0]][col + na[1]][0]
    nb_magnitude = gradient_array[row + nb[0]][col + nb[1]][0]
    if(gradient_magnitude > na_magnitude and gradient_magnitude > nb_magnitude):
        return True
    else:
        return False


def dive_in(gradient_array, visited_matrix, row, col, high_threshold, low_threshold):
    
    # Obtain the neighbouring pixels
    neighbours = obtain_neighbours(gradient_array, visited_matrix, row, col)
    # Check if any of the neighbours is already an edge or connected to one
    for neighbour in neighbours:
        if(neighbour[0] >= high_threshold or visited_matrix[neighbour[1]][neighbour[2]] == 2):
            visited_matrix[neighbour[1]][neighbour[2]] = 2
            visited_matrix[row][col] = 2
            return visited_matrix
        elif(neighbour[0] <= low_threshold):
            visited_matrix[neighbour[1]][neighbour[2]] = 1

    # Recursively dive_in from each neighbour to check if one of the medium threshold neighbours is an edge or not
    # Temporarily freeze the current pixel, visited_matrix[row][col] = 1, so no loop-back
    visited_matrix[row][col] = 1
    for neighbour in neighbours:
        nrow = neighbour[1]
        ncol = neighbour[2]
        if(neighbour[0] > low_threshold and visited_matrix[nrow][ncol] != 1):
            visited_matrix = dive_in(gradient_array, visited_matrix, nrow, ncol, high_threshold, low_threshold)
            if(visited_matrix[nrow][ncol] == 2):
                visited_matrix[row][col] = 2
                return visited_matrix
    return visited_matrix


def obtain_neighbours(gradient_array, visited_matrix, row, col):
    
    attempts = []
    neighbours = []
    filtered_neighbours = []

    attempts.append(obtain_neighbour(gradient_array, row + 0, col + 1))
    attempts.append(obtain_neighbour(gradient_array, row - 1, col + 1))
    attempts.append(obtain_neighbour(gradient_array, row - 1, col + 0))
    attempts.append(obtain_neighbour(gradient_array, row - 1, col - 1))
    attempts.append(obtain_neighbour(gradient_array, row + 0, col - 1))
    attempts.append(obtain_neighbour(gradient_array, row + 1, col - 1))
    attempts.append(obtain_neighbour(gradient_array, row + 1, col + 0))
    attempts.append(obtain_neighbour(gradient_array, row + 1, col + 1))

    for attempt in attempts:
        if(attempt != None):
            neighbours.append(attempt)

    for neighbour in neighbours:
        if(visited_matrix[neighbour[1]][neighbour[2]] != 1):
            filtered_neighbours.append(neighbour)
    
    return filtered_neighbours


def apply_hysteresis_thresholding(gradient_array, result_array):
    
    # 0 means unknown
    # 1 means below low thershold
    # 2 means it is marked as an edge

    visited_matrix = np.zeros((gradient_array.shape[0], gradient_array.shape[1]), dtype=np.int)
    high_threshold = 14
    low_threshold = 4
    for row in range(0, len(gradient_array)):
        for col in range(0, len(gradient_array[0])):
            if(gradient_array[row][col][0] >= high_threshold):
                visited_matrix[row][col] = 2        # actual edge
            elif(gradient_array[row][col][0] <= low_threshold):
                result_array[row][col] = 0
                visited_matrix[row][col] = 1        # below low
            elif(gradient_array[row][col][0] < high_threshold):
                visited_matrix = dive_in(gradient_array, visited_matrix, row, col, high_threshold, low_threshold)
                if(visited_matrix[row][col] == 1):
                    result_array[row][col] = 0
    return result_array

                
def canny_edge_detector(image, sigma):

    # Convert to numpy array
    image_array = np.asarray(image)
    result_array = np.zeros((image_array.shape[0], image_array.shape[1]), dtype=np.int)

    # Smooth image by Gaussian filter
    smoothed_image_array = gc.gaussconvolve2d(image_array, sigma)

    # Compute derivative of filtered image
    dx = cid.xderivative(smoothed_image_array.copy())
    dy = cid.yderivative(smoothed_image_array.copy())

    # Find magnitude and orientation of gradient
    row = []
    gradient_array = []
    # Match up the rows
    for ix, iy in zip(dx, dy):
        # For each row, match up columns
        for ixx, iyy in zip(ix, iy):
            gradient_magnitude = math.sqrt((math.pow(ixx,2) + math.pow(iyy,2)))
            gradient_direction = np.arctan2(iyy, ixx)
            row.append(np.asarray([gradient_magnitude, gradient_direction]))
        gradient_array.append(row)
        row = []
    gradient_array = np.asarray(gradient_array)

    # Apply non-maximum suppression
    for row in range(0, len(gradient_array)):
        for col in range(0, len(gradient_array[0])):
            pixel = gradient_array[row][col]
            if(is_maximum(row, col, pixel, gradient_array)):
                result_array[row][col] = 120

    # Apply hysteresis thresholding
    result_array = apply_hysteresis_thresholding(gradient_array, result_array)
    return result_array


image = Image.open('/.../parthenon.jpg')
image = image.convert('L')
result = canny_edge_detector(image, 2)
result = result.astype('uint8')
image = Image.fromarray(result)
image.save('/.../canny_edge_detection_parthenon.jpg','JPEG')