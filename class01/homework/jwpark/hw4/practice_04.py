import numpy as np
import cv2

img = cv2.imread("Lenna.png")
print(img.shape)

img = np.expand_dims(img, axis=0)
print(img.shape)

img = img.transpose(0, 3, 2, 1)
print(img.shape)
