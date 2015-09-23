from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import math
import scipy as sc
import ncc

#creates a pyramid for an image. It returns a list including
#the original PIL image followed by all the PIL images of 
#reduced size, using a scale factor of 0.75 from one level
# to the next. The pyramid should stop when any further
#reduction in size will make a dimension of the image smaller 
#than minsize. 

new_width =15

def MakePyramid(image, minsize):
    im_size   = image.size
    im_width  = im_size[0]
    im_height = im_size[1]
    
    min_width = minsize[0] 
    min_height= minsize[1]
    
    img_list  = [image]
      
    while (im_width>=min_width and im_height>=min_height):
	#resize the image, using the current values for im_width and im_height
        im2 = image.resize((int(im_width*0.75),int(im_height*0.75)), Image.BICUBIC)
		
	#add this resized image to the image array
        img_list.append(im2)
                            
	#scale the values of im_width and im_height for the next iteration
        im_width  = im_width*0.75
        im_height = im_height*0.75
    
    return img_list
    
def ShowPyramid(pyramid):
    image = Image.new("L", (1980,1080))
    offset_x=0
    offset_y=0
    n=len(pyramid)
	
	#loop over all the layers of the pyramid
    for i in range (0, n):
        im = pyramid[i]
        im = im.convert("L")
        
        im_size = im.size
        im_width= im_size[0]
        im_height= im_size[1]
        
	#paste the image at its position on the background
        image.paste(im,(offset_x,offset_y))
	#update the offsets for the next image to be placed at
        offset_x= offset_x + im_width
        offset_y= offset_y + im_height
    
    #image.save('pyramid','PNG')
    #image.show()
    arr = np.asarray(image)
    plt.imshow(arr, cmap = cm.Greys_r)
    plt.show()
    
def FindTemplate(pyramid, template, threshold, filename):
    scaleFactor = 0.75
    currentScale =1
    result = []
    
    #resize the template
    tmp_width= template.size[0]
    tmp_height= template.size[1] 

    tmp_len   = tmp_height*new_width/tmp_width
    template2 = template.resize((int(new_width),int(tmp_len)),Image.BICUBIC) 
        
    #use the template    
    for image in pyramid: 
        #fetch image
        image = image.convert('L')
        #array of cross-correlation coefficients
        arr= ncc.normxcorr2D(image, template2)       
        print arr 
 
	#compare each pixels correlation to the threshold
        for i in range (len(arr)):
            for j in range (len(arr[i])):
                if arr[i][j] >= threshold:
                    result.append((j*currentScale, i*currentScale,currentScale))
        currentScale *= 1/scaleFactor 
    
    # image to plot the detections on
    final = pyramid[0] 
    # convert it to be able to plot boxes in red
    final = final.convert('RGB')  
	
    for i in range(len(result)): 
	
	#the scale to multiply the coordinates by to 
	#have them display in the right place on the 
	#largest image in the pyramid	
        corrscale = result[i][2]  
		
        draw = ImageDraw.Draw(final)    
        x1 = (result[i][0]) + (tmp_width/2)*corrscale
        x2 = (result[i][0]) - (tmp_width/2)*corrscale
        y1 = (result[i][1]) + (tmp_height/2)*corrscale
        y2 = (result[i][1]) - (tmp_height/2)*corrscale  
		
        draw.line((x1,y1,x2,y1),fill="red") 
        draw.line((x1,y2,x2,y2),fill="red") 
        draw.line((x1,y1,x1,y2),fill="red") 
        draw.line((x2,y1,x2,y2),fill="red") 
    
    del draw    
    final.save(filename + '.png','PNG')                                                     

#open the template image
tmp = Image.open('C:/.../faces/template.jpg')

#open the three images
im1 = Image.open('C:/.../faces/judybats.jpg')
im2 = Image.open('C:/.../faces/students.jpg')
im3 = Image.open('C:/.../faces/tree.jpg')

#calculate their respective minimum width and height
im1_width = im1.size[0]
im1_height= im1.size[1]

im2_width = im2.size[0]
im2_height= im2.size[1]

im3_width = im3.size[0]
im3_height= im3.size[1]

min_width = 15
min_height= im1_height*(15/im1_width)
minsize1 = (min_width, min_height) 

min_height= im1_height*(15/im2_width)
minsize2 = (min_width, min_height) 

min_height= im1_height*(15/im3_width)
minsize3 = (min_width, min_height) 

#compute their pyramid representations
pyramid1 = MakePyramid(im1, minsize1)
pyramid2 = MakePyramid(im2, minsize2)
pyramid3 = MakePyramid(im3, minsize3)

#display the pyramids
ShowPyramid(pyramid1)
ShowPyramid(pyramid2)
ShowPyramid(pyramid3)

#plot the template detections
FindTemplate(pyramid1, tmp, 0.52, 'jbats')
FindTemplate(pyramid2, tmp, 0.52, 'students')
FindTemplate(pyramid3, tmp, 0.52, 'tree')
