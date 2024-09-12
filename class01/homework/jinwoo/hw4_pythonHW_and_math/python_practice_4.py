


import cv2
import numpy as np
import matplotlib.pylab as plt

image = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)

expended_image = np.expand_dims(image, axis=0)
transposed_expImage = np.transpose(expended_image)

print("원본: ", image.shape)
print("확장: ", expended_image.shape)
print("변환: ", transposed_expImage.shape)
