# 실습5
# 1. 붉은색 네모 박스를 직접 구현.
# 붉은색 네모 박스 내용
# output = cv2.filter2D(img, -1, kernel)

import cv2
import numpy as np

img = cv2.imread('lama.png', cv2.IMREAD_GRAYSCALE)
f_identity = np.array([[0, 0, 0],[0, 1, 0],[0, 0, 0]])
f_ridge = np.array([[0, -1, 0],[-1, 4, -1],[0, -1, 0]])
f_edgeDetection = np.array([[-1, -1, -1],[-1, 8, -1],[-1, -1, -1]])
f_sharpen = np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]])
f_boxBlur = np.array([[1, 1, 1],[1, 1, 1],[1, 1, 1]])*(1/9)
f_gaussianBlue_3x3 = np.array([[1, 2, 1],[2, 4, 2],[1, 2, 1]])*(1/16)
f_gaussianBlue_5x5 = np.array([[1, 4, 6, 4, 1],[4, 16, 24, 16, 4],[6, 24 ,36, 24, 6],[4, 16, 24, 16, 4],[1, 4, 6, 4, 1]])*(1/256)
print(f_ridge)
output = cv2.filter2D(img, -1, f_ridge)
cv2.imshow('edge', output)
cv2.waitKey(0)
