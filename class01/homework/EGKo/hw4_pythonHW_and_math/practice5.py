import cv2
import numpy as np

img = cv2.imread("./image/cat.jpg", cv2.IMREAD_GRAYSCALE)

kernel = np.array([[1, 1, 1],
                   [1, -8, 1],
                   [1, 1, 1]])

rows, cols = img.shape
pad_img = np.zeros((rows + 2, cols + 2), dtype=np.uint8)
pad_img[1:-1, 1:-1] = img

output = np.zeros_like(img, dtype=np.int32)

for row in range(1, rows + 1):
    for col in range(1, cols + 1):
        region = pad_img[row-1:row+2, col-1:col+2]
        output[row-1, col-1] = np.sum(region * kernel)

output = np.clip(output, 0, 255)
output = output.astype(np.uint8)

cv2.imshow('Original Image', img)
cv2.imshow('Filtered Image', output)
cv2.waitKey(0)
cv2.destroyAllWindows()