#%%
import cv2
import numpy as np

img = cv2.imread('lama.jpg', cv2.COLOR_BGR2RGB)
kernel = np.array([[1, 4, 6, 4, 1], 
                   [4, 16, 24, 16, 4], 
                   [6, 24, 36, 24, 6], 
                   [4, 16, 24, 16, 4], 
                   [1, 4, 6, 4, 1]])/256
print(kernel)
output = cv2.filter2D(img, -1, kernel)

cv2.imshow('edge', output)
cv2.waitKey(0)