from PIL import Image
import numpy as np
import math
from scipy import signal

def xderivative(image):  
    # PIL and numpy use different internal representations
    # convert the image to a numpy array (for subsequent processing)
    image_array = np.asarray(image)
    # Note: we need to make a copy to change the values of an array created using np.asarray
    image_array_copy = image_array.copy()

    xfilter = [1, -1]

    for row in range(0, image.size[1]):
        for col in range(0, image.size[0]):
            if(col >= math.floor(len(xfilter)/2) and col <= image.size[0] - math.floor(len(xfilter)/2) - 1):
                delta = (image_array_copy[row, col] * xfilter[0]) + (image_array_copy[row, col + 1] * xfilter[1])
                if(delta > 50):
                    image_array_copy[row, col] = delta
                else:
                    image_array_copy[row, col] = 0
            else:
                image_array_copy[row, col] = 256

    result = Image.fromarray(image_array_copy)
    return result

def yderivative(image):  
    # PIL and numpy use different internal representations
    # convert the image to a numpy array (for subsequent processing)
    image_array = np.asarray(image)
    # Note: we need to make a copy to change the values of an array created using np.asarray
    image_array_copy = image_array.copy()

    xfilter = [1, -1]

    for row in range(0, image.size[1]):
        for col in range(0, image.size[0]):
            if(row >= math.floor(len(xfilter)/2) and row <= image.size[1] - math.floor(len(xfilter)/2) - 1):
                delta = (image_array_copy[row, col] * xfilter[0]) + (image_array_copy[row + 1, col] * xfilter[1])
                if(delta > 50):
                    image_array_copy[row, col] = delta
                else:
                    image_array_copy[row, col] = 0
            else:
                image_array_copy[row, col] = 256

    result = Image.fromarray(image_array_copy)
    return result

### Image Derivatives ###
image = Image.open('.../perast.jpg')
# convert the image to a black and white "luminance" greyscale image
image = image.convert('L')
# convert the result back to a PIL image and save
result = xderivative(image)
result.save('/.../xresultm.jpg','JPEG')
result = yderivative(image)
result.save('/.../yresultm.jpg','JPEG')
