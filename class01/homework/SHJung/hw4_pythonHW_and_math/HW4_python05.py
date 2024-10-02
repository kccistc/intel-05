import cv2
import numpy as np

img = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)
kernel = np.array([[1, 1, 1],[1, -8, 1], [1, 1, 1]])
print(kernel)

output_auto = cv2.filter2D(img, -1, kernel)

height, width = img.shape

output_manual = np.zeros_like(img, dtype = np.float32)

padded_img = np.pad(img, ((1, 1), (1, 1)))

for i in range(1, height + 1):
    for j in range(1, width + 1):
        region = padded_img[i-1:i+2, j-1:j+2] 
        output_manual[i-1, j-1] = np.sum(region * kernel) 

output = np.clip(output_manual, 0, 255).astype(np.uint8)

combined_img = cv2.hconcat([output_auto, output_manual])
cv2.imshow('edge', combined_img)
cv2.waitKey(0)
