import cv2
import numpy as np

path = r'C:\imagens display\IMG_30.jpg'

image = cv2.imread(path)
width = 1080
height = 920

# dsize
dsize = (width, height)

# resize image
output = cv2.resize(image, dsize)

kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
sharpened = cv2.filter2D(output, -1, kernel) # applying the sharpening kernel to the input image & displaying it.
cv2.imshow('Image Sharpening', sharpened)
cv2.waitKey(0)
cv2.destroyAllWindows()
