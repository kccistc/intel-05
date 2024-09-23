# 실습5
# 1. 붉은색 네모 박스를 직접 구현.
# 붉은색 네모 박스 내용
# output = cv2.filter2D(img, -1, kernel)

import cv2
import numpy as np

img = cv2.imread('flower.jpg', cv2.IMREAD_GRAYSCALE)
kernel = np.array([[1, 1, 1],[1, -8, 1],[1, 1, 1]])
print(kernel)
output = cv2.filter2D(img, -1, kernel)
cv2.imshow('edge', output)
cv2.waitKey(0)