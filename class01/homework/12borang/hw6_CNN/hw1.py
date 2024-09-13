# %%
import cv2
import numpy as np

img = cv2.imread('lena.png', cv2.COLOR_BGR2RGB)

kernel = np.array([[1, 4, 6, 4, 1],
                   [4, 16, 24, 16, 4],
                   [6, 24, 36, 24, 6],
                   [4, 16, 24, 16, 4],
                   [1, 4, 6, 4, 1]], dtype=np.uint8) / 256

output = cv2.filter2D(img, -1, kernel)

output_clipped = np.clip(output, 0, 255).astype(np.uint8)

cv2.imshow('Filtered Image', output_clipped)
cv2.waitKey(0)
cv2.destroyAllWindows()