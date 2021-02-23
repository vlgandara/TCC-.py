import cv2
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox

path = r'C:\imagens display\IMG_3.jpg'
im = cv2.imread(path)

img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) 

bbox, label, conf = cv.detect_common_objects(img_rgb)

print(bbox, label, conf)

output_image = draw_bbox(img_rgb, bbox, label, conf)

plt.imshow(output_image)
plt.show()

