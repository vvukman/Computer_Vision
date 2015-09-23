from PIL import Image
import numpy as np
import math
from scipy import signal

def boxfilter(n):
    #Check if the given value is odd
    assert ((n-1)%2==0), "Dimension must be odd."
	#Create Numpy array
    x = (1.0/(n*n))*np.ones((n,n),dtype=np.double)
	#If must specifically be a numpy array, this works too:
	#x = (1.0/(n*n))*np.array(np.ones((n,n),dtype=np.double),dtype=np.double) 
    print x

def gauss1d(sigma):
	#Multiply sigma by 6, round it to next closest integer
    length = int(math.ceil(6*sigma))
	#If length is even, increment it by one to make it odd
    if(length%2==0): length = length+1
    
    center = (length+1)/2              # center index
    lep = ((-1)*center)+1              # left endpoint
    rep = center                       # right endpoint
    arr = np.arange(lep, rep)          # create array of distances from center
    
    arr  = arr.astype(float)
    arr  = np.exp(-((pow(arr,2)))/(2*(pow(sigma,2))))  # the gaussian approximation
    norm = np.sum(arr)                                 # the value to normalize by
    gaus = arr/norm                                    # the normalized version of x       
    return gaus                                                                                                                                                      

def gauss2d(sigma):
    first = gauss1d(sigma)             #the first 1d gaussian filter
    first = first[np.newaxis]          #make it a 2d array
    b = np.transpose(first)            #compute the transpose
    conv=signal.convolve2d(first, b)   #convolve first with its transpose
    return conv

def gaussconvolve2d(array, sigma):
    filt = gauss2d(sigma)                
    newg = signal.convolve2d(array, filt,'same')
    return newg

#Apply filtering to jaguar image
  
#open the test image
im = Image.open('C:/.../fence.png')
#print dimensions
print im.size, im.mode, im.format
#convert image to grey scale
im=im.convert('L')
im.save('fence_greyscale','PNG')
#convert image to numpy array
im_array=np.asarray(im)
im2_array=im_array.copy()
#call convolve2d
im3_array=gaussconvolve2d(im2_array, 3)
#convert array to uint8 format
im3_array=im3_array.astype('uint8')
#convert back to image
im3=Image.fromarray(im3_array)
#save image
im3.save('C:/.../fence_blur','PNG')