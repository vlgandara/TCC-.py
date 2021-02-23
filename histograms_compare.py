import cv2
import numpy as np

image_1_path = r'C:\imagens display\IMG_0R.jpg'
image_2_path = r'C:\imagens display\IMG_20.jpg'

class CompareImage(object):

    def __init__(self, image_1_path, image_2_path):
        self.minimum_commutative_image_diff = 1
        self.image_1_path = image_1_path
        self.image_2_path = image_2_path

    def compare_image(self):
        window_name_1 = 'image_1'
        window_name_2 = 'image_2'
        image_1 = cv2.imread(self.image_1_path, 0)
        image_2 = cv2.imread(self.image_2_path, 0)
        width = 400
        height = 400

        # dsize
        dsize = (width, height)

        # resize image
        output1 = cv2.resize(image_1, dsize)
        output2 = cv2.resize(image_2, dsize)
        
        
        cv2.imshow(window_name_1, output1)
        cv2.imshow(window_name_2, output2)

        commutative_image_diff = self.get_image_difference(image_1, image_2)

        if commutative_image_diff < self.minimum_commutative_image_diff:
            print ("Matched")
            return commutative_image_diff
        print ("Imagens diferentes")
        return commutative_image_diff

    @staticmethod
    def get_image_difference(image_1, image_2):
        first_image_hist = cv2.calcHist([image_1], [0], None, [256], [0, 256])
        second_image_hist = cv2.calcHist([image_2], [0], None, [256], [0, 256])

        img_hist_diff = cv2.compareHist(first_image_hist, second_image_hist, cv2.HISTCMP_BHATTACHARYYA)
        img_template_probability_match = cv2.matchTemplate(first_image_hist, second_image_hist, cv2.TM_CCOEFF_NORMED)[0][0]
        img_template_diff = 1 - img_template_probability_match

        # apenas 10% da diferença do histograma
        commutative_image_diff = (img_hist_diff / 10) + img_template_diff
        return commutative_image_diff


if __name__ == '__main__':
    compare_image = CompareImage(image_1_path, image_2_path)
    image_difference = compare_image.compare_image()
    print (image_difference)
    
