import cv2
import numpy as np

img = cv2.imread("./image/cat.jpg", cv2.IMREAD_GRAYSCALE)
kernel = np.array([[1,1,1],[1,-8,1],[1,1,1]])
print(kernel)
print(img.shape)
rows, cols = img.shape
pad = np.zeros([rows+2, cols+2])
pad_img = pad
print(pad.shape)
print(rows)
print(cols)
# for row in range(1,rows-1):
#     for col in range(1,cols-1):
#         pannel[row, col] += img[row, col]


output = cv2.filter2D(img, -1, kernel)
cv2.imshow('pannel', pannel)
cv2.imshow('edge', output)
cv2.waitKey(0)