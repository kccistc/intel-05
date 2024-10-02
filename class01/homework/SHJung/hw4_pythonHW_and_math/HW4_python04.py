import cv2
import numpy as np

image = cv2.imread('Lenna.png')
image = np.expand_dims(image, axis = 0)

print(image.shape)