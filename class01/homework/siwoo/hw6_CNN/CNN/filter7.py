import cv2
import numpy as np

img = cv2.imread('lama.png', cv2.IMREAD_COLOR)

kernel = np.array([[1, 4, 6, 4, 1], [4, 6, 24,16,4], [6, 24, 36,24,6], [4,16,24,16,4],[1, 4, 6, 4, 1]], dtype=float) / 256.0

output = cv2.filter2D(img, -1, kernel)
cv2.imshow('edge', output)
cv2.waitKey(0)

