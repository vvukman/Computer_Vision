from PIL import Image, ImageDraw
import numpy as np
import csv
import math
import matplotlib.pyplot as plt  # I needed to add these packages to view output
import matplotlib.cm as cm  # I needed to add these packages to view output

def ReadKeys(image):
    """Input an image and its associated SIFT keypoints.

    The argument IMAGE is the image file name (without an extension).
    The image is read from the PGM format file image.pgm and the
    KEYPOINTS are read from the file image.key.

    ReadKeys returns the following 3 arguments:

    IMAGE: the image (in PIL 'RGB' format)

    KEYPOINTS: K-by-4 array, in which each row has the 4 values specifying
    a KEYPOINT (row, column, scale, orientation).  The orientation
    is in the range [-PI, PI] radians.
	K-by-4 because we can have K KEYPOINTS and each has 4 dimensions. 

    DESCRIPTORS: a K-by-128 array, where each row gives a descriptor
    for one of the K KEYPOINTS. The descriptor is a 1D array of 128
    values with unit length.
	
	# We get an image. Open it. CSV file. GO row by row. If row is of length
	# of 4, then we have an interest point with 4 fields. Add it to our 
	# keypoints array. A new interest point with 4 fields. 
	
	# Images can have many keypoints and each keypoint has a descriptor. 
	# It is for a 16x16 neighbourhood around the pixel. The 16x16 is 
	# divided into 16 4x4 regions. For each 4x4, we get the gradient 
	# magnitude and direction of the pixels. We divide the 360 degree 
	# space into 8 bins of 45 degrees each. We tally up the directions 
	# of each of the 4x4 regions. There will be 16 blocks of 4x4s, with
	# each having 8 dimensions. 
	
	# So if the length of a row is 8, it is an 8 dimensional space. It
	# is the histogram for a 4x4 neighbourhood around the interest point. 
	
	
    """
    im = Image.open(image+'.pgm').convert('RGB')
    keypoints = []
    descriptors = []
    first = True
    with open(image+'.key','rb') as f:
        reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC,skipinitialspace = True)
        descriptor = []
        for row in reader:
            if len(row) == 2:
                assert first, "Invalid keypoint file header."
                assert row[1] == 128, "Invalid keypoint descriptor length in header (should be 128)."
                count = row[0]
                first = False
            if len(row) == 4:
                keypoints.append(np.array(row))
            if len(row) == 20:
                descriptor += row
            if len(row) == 8:
                descriptor += row
                assert len(descriptor) == 128, "Keypoint descriptor length invalid (should be 128)."
                #normalize the key to unit length
                descriptor = np.array(descriptor)
                descriptor = descriptor / math.sqrt(np.sum(np.power(descriptor,2)))
                descriptors.append(descriptor)
                descriptor = []
    assert len(keypoints) == count, "Incorrect total number of keypoints read."
    print "Number of keypoints read:", int(count)
    return [im,keypoints,descriptors]

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

def DisplayMatches(im1, im2, matched_pairs):
    """Display matches on a new image with the two input images placed side by side.

    Arguments:
     im1           1st image (in PIL 'RGB' format)
     im2           2nd image (in PIL 'RGB' format)
     matched_pairs list of matching keypoints, im1 to im2

    Displays and returns a newly created image (in PIL 'RGB' format)
    """
    im3 = AppendImages(im1,im2)
    offset = im1.size[0]
    draw = ImageDraw.Draw(im3)
    for match in matched_pairs:
        draw.line((match[0][1], match[0][0], offset+match[1][1], match[1][0]),fill="red",width=2)
    arr = np.asarray(im3)
    plt.imshow(arr, cmap = cm.Greys_r)
    plt.show()
    return im3

def match(image1,image2):
    """Input two images and their associated SIFT keypoints.
    Display lines connecting the first 5 keypoints from each image.
    Note: These 5 are not correct matches, just randomly chosen points.

    The arguments image1 and image2 are file names without file extensions.

    Returns the number of matches displayed.

    Example: match('scene','book')
    """
    im1, keypoints1, descriptors1 = ReadKeys(image1)
    im2, keypoints2, descriptors2 = ReadKeys(image2)
    #
    # REPLACE THIS CODE WITH YOUR SOLUTION (ASSIGNMENT 5, QUESTION 3)
    #
	
	# Each image is going to have interest points and keypoint descriptors. 
	# We are going to go through each of the keypoints for im1, and for each 
	# keypoint, we are going to take its descriptor and iterate over all the 
	# keypoints' descriptors in im2. For each descriptor, we compute its distance
	# from the current descriptor from im1 using its angle. 
	
	# Each keypoint is going to have 128 dimensional descriptor. 
	
    angles = []
    matches = []
    votes = []
    result = []
	# for each interest point in first image...
    for kp1 in range (len(keypoints1)):
	# take its corresponding descriptor...
	d1 = descriptors1[kp1]
	# iterating over all the interest points in the second image...
        for kp2 in range (len(keypoints2)):
	   # take the descriptor for the current interest point we are at...
           d2 = descriptors2[kp2]
	   # compute dot product ...
	   angle = math.acos(np.dot(d1,d2))
	   # append the dot product ...
 	   angles.append(angle)
	   # when we are dont computing all of the dot products for this IP,
	   # get the best match for this interest point. 
        # the array containing original indexes of sorted values...
        indexes = sorted(range(len(angles)),key=lambda x:angles[x])
        print len(indexes)
	# take the best one for now...
	matchindex1 = indexes[0]
	matchindex2 = indexes[1]
	print[matchindex1]
	print[matchindex2]
	ratio = float(angles[matchindex1])/float(angles[matchindex2])
	print ratio
	if (ratio<0.70):
	   # extract that keypoint...
	   kpmatch = keypoints2[matchindex1]
	   matches.append([keypoints1[kp1], kpmatch])
	del angles[:]

    for r in range (10):
	rand = matches[np.random.randint(len(matches))]
	# Check that the change of orientation between the two keypoints of 
	# each match is within 30 degrees.
		
	kp1 = rand[0]
	kp2 = rand[1]
		
	changeScale1 = abs(kp1[2] - kp2[2])
	changeOrientation1 = kp1[3] - kp2[3]
		
	# Check other matches for consistency
	for match in matches:
		m1 = match[0]	# keypoint from image 1
		m2 = match[1]	# keypoint from image 2
	
		changeScale2 = abs(m1[2] - m2[2])         #difference in scale
		changeOrientation2 = m1[3] - m2[3]   #difference in orientation
		
		# Check consistency with respect to orientation (radians)
		# We want the absolute difference in degrees between the two angles.
		# Convert to degrees. Because orientations are equal modulo 180 degrees, 
		# take modulo at the end. 
		DD1 = math.degrees(changeOrientation1) + 360
		DD2 = math.degrees(changeOrientation2) + 360
		difference = abs(DD1 - DD2) % 180
		if (difference > 20):
		  # Does not satisfy consistency constraint.
		  continue				
			
		# Now check if consistent with respect to scales. 
		Smax = max(changeScale1, changeScale2)
		Smin = min(changeScale1, changeScale2)
		# We want the scales to be within 50% of each other. 
		th = float(Smax * 0.30)
		
		if (th > Smin):
		  # Does not satisfy consistency constraint.
		  continue	
		votes.append(match)
		
	if (len(votes) > len(result)):
	   result = votes			
    
    #Generate five random matches (for testing purposes)
    matched_pairs = []
    for i in range(len(result)):
        matched_pairs.append([result[i][0], result[i][1]])
    #
    # END OF SECTION OF CODE TO REPLACE
    #
    im3 = DisplayMatches(im1, im2, matched_pairs)
    return im3

#Test run...
match('C:/.../library','C:/.../library2')

