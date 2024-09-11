import cv2
import numpy as np

img = cv2.imread('lena.png')
print(img.shape)
img = np.expand_dims(img, axis = 0)
print(img.shape)
img = img.transpose(0,3,2,1)
print(img.shape)