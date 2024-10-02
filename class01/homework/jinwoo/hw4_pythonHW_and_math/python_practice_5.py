import cv2
import numpy as np

img = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)
kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
print(kernel)
output_autoFiltered = cv2.filter2D(img, -1, kernel)
output_manualFiltered = np.zeros_like(img)

h, w = img.shape
kh, kw = kernel.shape

for i in range(1, h - 1):
    for j in range(1, w - 1):
        sample = img[i - 1:i + 2, j - 1:j + 2]
        output_manualFiltered[i, j] = np.clip(np.sum(sample * kernel), 0, 255)

cv2.imshow('original', img)
cv2.imshow('autoFiltered', output_autoFiltered)
cv2.imshow('manualFiltered', output_manualFiltered)
cv2.waitKey(0)
cv2.destroyAllWindows()