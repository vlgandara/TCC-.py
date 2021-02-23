import numpy as np
import cv2
from matplotlib import pyplot as plt

image_1_path = r'C:\imagens display\IMG_15.jpg'
image_2_path = r'C:\imagens display\iMG_18.jpg'

img1 = cv2.imread(image_1_path) 
img2 = cv2.imread(image_2_path) 

#sift
sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

print(len(des1))
print(len(des2))

#flann 
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=100)   

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1,des2,k=2)

matchesMask = [[0,0] for i in range(len(matches))]

good = []
total = []
for i,(m,n) in enumerate(matches):
    total.append([i])
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
        good.append([m])

print(len(good))

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

plt.imshow(img3,),plt.show()
