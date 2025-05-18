import cv2
import numpy as np
import matplotlib.pyplot as plt
# Load and preprocess image

def add_checkerboard(image, center, size, width):
    #adds a checkerboard pattern to an existing image by changing the pixel coloring
    #center is the center location of the checkerboard
    #width is the number of pixels between checkboard squares
    #size is the size of each checkerboard square

    xc, yc = center.astype(int) 
    xc_og = xc
    fill_white = np.array([255, 255, 255])
    fill_black = np.array([0, 0, 0])
    step = 1
    for j in range(0,width,step):
        xc = xc_og
        for i in range(0,width,step):
            if (i//step + j//step) % 2 == 0:
                image[yc-size//2:yc+size//2, xc-size//2:xc+size//2,:] = fill_black
            else:
                image[yc-size//2:yc+size//2, xc-size//2:xc+size//2,:] = fill_white
            xc += size*step
        yc+= size*step
    return image
    
    
    

image = cv2.imread('far.jpg')
h,w,_= image.shape
center = (np.array([h,w]) // 2)
print(center)
image = add_checkerboard(image, center, 11, 10)

pattern_size = (9,9)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
print(corners)
# Draw corners if found
if ret:
    cv2.drawChessboardCorners(image, pattern_size, corners, ret)
    print("Checkerboard detected.")
else:
    print("Checkerboard not detected.")

# Show result
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Checkerboard Detection')
plt.axis('off')
plt.show()


    

# # image[h/2,w/2,:] = np.array([0, 0, 255])


# cv2.imshow("Crater Detection (No Canny)", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()