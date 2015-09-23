from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import signal
import numpy.linalg as lin

# START OF FUNCTIONS CARRIED FORWARD FROM ASSIGNMENT 2

def boxfilter(n):
    #Check if the given value is odd
    assert ((n-1)%2==0), "Dimension must be odd."
	#Create Numpy array
    x = (1.0/(n*n))*np.ones((n,n),dtype=np.double)
	#If must specifically be a numpy array, this works too:
	#x = (1.0/(n*n))*np.array(np.ones((n,n),dtype=np.double),dtype=np.double) 
    return x

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

# END OF FUNCTIONS CARRIED FORWARD FROM ASSIGNMENT 2

# Convolve an image with a boxfilter of size n. Used in function below.
def boxconvolve2d(image, n):
    bfilter = boxfilter(n)
    result = signal.convolve2d(image, bfilter, 'same')
    return result	

# Estimate spatial derivatives of im1 and temporal derivative from im1 to im2.
def Estimate_Derivatives(im1, im2, sigma=1.5, n=3):  
  
    # Smooth im1 with a 2D Gaussian of the given sigma.  
    im1_smoothed = gaussconvolve2d(im1, sigma)
	
    #Use first central difference to estimate derivatives.
    Ix, Iy = np.gradient(im1_smoothed)	# function gives an array for each dimension's derivative.
	
    # Use point-wise difference between (boxfiltered) im2 and im1 to estimate temporal derivative
    It = boxconvolve2d(im2, n) - boxconvolve2d(im1, n)
	
    #return the three spatial derivatives. 
    return Ix, Iy, It

def Optical_Flow(im1, im2, x, y, window_size, sigma=1.5, n=3):
	
    # Here we are going to apply the Lucas Kanade method. We have two images, at two temporal
    # frequencies. We also have a given pixel at an (x,y) coordinate. We have a window size 
    # which is centered at the current pixel of interest. Finally, there is a sigma value
    # for smoothing reasons and an n value.
	
    # im1 - the source image
    # im2 - the destination image
    # x,y - the location of the point to track in the source image
    # window_size - the size of the window to track
    # sigma - the sigma value to use when estimating derivatives of source and destination images
    # n - the box filter size to use when estimating derivatives of source and destination images
    
    assert((window_size % 2) == 1) , "Window size must be odd"

    # Obtain the 2 spatial derivatives for the one image and the temporal derivative.
    Ix, Iy, It = Estimate_Derivatives(im1, im2, sigma, n)
    half = np.floor(window_size/2)
    # select the three local windows of interest. 
    # y-half --> y+half+1 (where y+half+1 is not included)
    win_Ix = Ix[y-half:y+half+1, x-half:x+half+1].T
    win_Iy = Iy[y-half:y+half+1, x-half:x+half+1].T
    win_It = -It[y-half:y+half+1, x-half:x+half+1].T
	
    # Must stack the arrays using np.vstack.
    A = np.vstack((win_Ix.flatten(), win_Iy.flatten())).T
    V = np.dot(np.linalg.pinv(A), win_It.flatten())
    # change the return line to:
    return V[1], V[0]

def AppendImages(im1, im2):
    """Create a new image that appends two images side-by-side.

    The arguments, im1 and im2, are PIL images of type RGB
    """
    im1cols, im1rows = im1.size
    im2cols, im2rows = im2.size
    im3 = Image.new('RGB', (im1cols+im2cols, max(im1rows,im2rows)))
    im3.paste(im1,(0,0))
    im3.paste(im2,(im1cols,0))
    return im3

def DisplayFlow(im1, im2, x, y, uarr, varr):
    """Display optical flow match on a new image with the two input frames placed side by side.

    Arguments:
     im1           1st image (in PIL 'RGB' format)
     im2           2nd image (in PIL 'RGB' format)
     x, y          point coordinates in 1st image
     u, v          list of optical flow values to 2nd image

    Displays and returns a newly created image (in PIL 'RGB' format)
    """
    im3 = AppendImages(im1,im2)
    offset = im1.size[0]
    draw = ImageDraw.Draw(im3)
    xinit = x+uarr[0]
    yinit = y+varr[0]
    for u,v,ind in zip(uarr[1:], varr[1:], range(1, len(uarr))):
		draw.line((offset+xinit, yinit, offset+xinit+u, yinit+v),fill="red",width=2)
		xinit += u
		yinit += v
    draw.line((x, y, offset+xinit, yinit), fill="yellow", width=2)
    im3.show()
    del draw
    return im3

def HitContinue(Prompt='Hit any key to continue'):
    raw_input(Prompt)

##############################################################################
#                  Here's your assigned target point to track                #
##############################################################################

# uncomment the next two lines if the leftmost digit of your student number is 0
#x=222
#y=213
# uncomment the next two lines if the leftmost digit of your student number is 1
#x=479
#y=141
# uncomment the next two lines if the leftmost digit of your student number is 2
#x=411
#y=242
# uncomment the next two lines if the leftmost digit of your student number is 3
#x=152
#y=206
# uncomment the next two lines if the leftmost digit of your student number is 4
#x=278
#y=277
# uncomment the next two lines if the leftmost digit of your student number is 5
#x=451
#y=66
# uncomment the next two lines if the leftmost digit of your student number is 6
#x=382
#y=65
# uncomment the next two lines if the leftmost digit of your student number is 7
x=196
y=197
# uncomment the next two lines if the leftmost digit of your student number is 8
#x=274
#y=126
# uncomment the next two lines if the leftmost digit of your student number is 9
#x=305
#y=164

##############################################################################
#                            Global "magic numbers"                          #
##############################################################################

# window size (for estimation of optical flow)
window_size=51

# sigma of the 2D Gaussian (used in the estimation of Ix and Iy)
sigma=1.5

# size of the boxfilter (used in the estimation of It)
n = 3

##############################################################################
#             basic testing (optical flow from frame 7 to 8 only)            #
##############################################################################

# scale factor for display of optical flow (to make result more visible)
scale=10

PIL_im1 = Image.open('C:/.../frame07.png')
PIL_im2 = Image.open('C:/.../frame08.png')
im1 = np.asarray(PIL_im1)
im2 = np.asarray(PIL_im2)
dx, dy = Optical_Flow(im1, im2, x, y, window_size, sigma, n)
print 'Optical flow: [', dx, ',', dy, ']'
plt.imshow(im1, cmap='gray')
plt.hold(True)
plt.plot(x,y,'xr')
plt.plot(x+dx*scale,y+dy*scale, 'dy')
print 'Close figure window to continue...'
plt.show()
uarr = [dx]
varr = [dy]

##############################################################################
#                   run the remainder of the image sequence                  #
##############################################################################

# UNCOMMENT THE CODE THAT FOLLOWS (ONCE BASIC TESTING IS COMPLETE/DEBUGGED)

print 'frame 7 to 8'
DisplayFlow(PIL_im1, PIL_im2, x, y, uarr, varr)
HitContinue()

print "Frame 7 tracked: (", x, ",", y, ")"

prev_im = im2
xcurr = x+dx
ycurr = y+dy
offset = PIL_im1.size[0]

print "Frame 8 tracked: (", xcurr, ",", ycurr, ")"

for i in range(8, 14):
    im_i = 'frame%0.2d.png'%(i+1)
    print 'frame', i, 'to', (i+1)
    PIL_im_i = Image.open('C:/.../%s'%im_i)
    numpy_im_i = np.asarray(PIL_im_i)
    dx, dy = Optical_Flow(prev_im, numpy_im_i, xcurr, ycurr, window_size, sigma, n)
    xcurr += dx
    ycurr += dy
    prev_im = numpy_im_i
    uarr.append(dx)
    varr.append(dy)
    # redraw the (growing) figure
    print "Frame",i+1, "tracked: (", xcurr, ",", ycurr, ")"
    DisplayFlow(PIL_im1, PIL_im_i, x, y, uarr, varr)
    HitContinue()
##############################################################################
# Don't forget to include code to document the sequence of (x, y) positions  #
# of your feature in each frame successfully tracked.                        #
##############################################################################
