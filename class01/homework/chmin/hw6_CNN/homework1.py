import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 로드
img1 = cv2.imread('lama.png')
img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

# 필터 리스트
kernels = [
    np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
    np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]),
    np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
    np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
    (1/9) * np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
    (1/16) * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]),
    (1/256) * np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]]),
    (-1/256) * np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, -476, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]])
]

# 필터를 적용하고 이미지를 서브플롯에 시각화
plt.figure(figsize=(12, 12))
plt.subplot(3, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

for i, kernel in enumerate(kernels, start=2):
    output = cv2.filter2D(img, -1, kernel)
    plt.subplot(3, 3, i)
    plt.imshow(output, cmap='gray')
    plt.title(f'Filtered {i-1}')
    plt.axis('off')

plt.tight_layout()
plt.show()
