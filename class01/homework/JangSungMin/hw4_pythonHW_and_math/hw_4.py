import cv2
import numpy as np


image_path = 'img.jpeg'
image = cv2.imread(image_path)

image_array = np.array(image)

expanded_image_array = np.expand_dims(image_array, axis=0)

transposed_image_array = np.transpose(expanded_image_array, (0, 3, 1, 2))

# 차원 확인
print('Original shape:', image_array.shape)
print('Expanded shape:', expanded_image_array.shape)
print('Transposed shape:', transposed_image_array.shape)
