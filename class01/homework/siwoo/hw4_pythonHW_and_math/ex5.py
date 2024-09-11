import cv2
import numpy as np

img = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)

kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])

image_height, image_width = img.shape
kernel_height, kernel_width = kernel.shape

pad_height = kernel_height // 2
pad_width = kernel_width // 2

padded_image = np.pad(img, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)

output = np.zeros_like(img)

for i in range(image_height):
    for j in range(image_width):
        roi = padded_image[i:i+kernel_height, j:j+kernel_width]
        
        filtered_value = np.sum(roi * kernel)
        
        output[i, j] = np.clip(filtered_value, 0, 255)

cv2.imshow('Edge Detection', output)
cv2.waitKey(0)
cv2.destroyAllWindows()
