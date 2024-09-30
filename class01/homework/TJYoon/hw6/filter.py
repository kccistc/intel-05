import cv2
import numpy as np
img = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)

kernel = np.array([[0,0,0],[0,1,0], [0,0,0]])
output = cv2.filter2D(img, -1, kernel)
cv2.imshow('Identity', output)

kernel = np.array([[0,-1,0],[-1,4,-1], [0,-1,0]])
output = cv2.filter2D(img, -1, kernel)
cv2.imshow('Ridge', output)

kernel = np.array([[0,-1,0],[-1,5,-1], [0,-1,0]])
output = cv2.filter2D(img, -1, kernel)
cv2.imshow('Sharpen', output)

kernel = np.array([[1,1,1],[1,1,1], [1,1,1]])/9
output = cv2.filter2D(img, -1, kernel)
cv2.imshow('Box blur', output)

kernel = np.array([[1,2,1],[2,4,2], [1,2,1]])/16
output = cv2.filter2D(img, -1, kernel)
cv2.imshow('Gaussian blur33', output)

kernel = np.array([[1,4,6,4,1],[4,16,24,16,4], [6,24,36,24,6], [4,16,24,16,4], [1,4,6,4,1]])/256
output = cv2.filter2D(img, -1, kernel)
cv2.imshow('Gaussian blur55', output)

cv2.waitKey(0)