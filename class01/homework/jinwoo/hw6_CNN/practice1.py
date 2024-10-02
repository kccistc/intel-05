import cv2
import numpy as np

img = cv2.imread('Lenna.png')
kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
output = cv2.filter2D(img, -1, kernel)

cv2.imshow('Identity', output)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%

kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
output = cv2.filter2D(img, -1, kernel)

cv2.imshow('Ridge detection', output)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%

kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
output = cv2.filter2D(img, -1, kernel)

cv2.imshow('Edge detection', output)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%

kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
output = cv2.filter2D(img, -1, kernel)

cv2.imshow('Sharpen', output)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%

kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9
output = cv2.filter2D(img, -1, kernel)

cv2.imshow('Box blur', output)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%

kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
output = cv2.filter2D(img, -1, kernel)

cv2.imshow('Gaussian blur 3x3', output)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%

kernel = np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]]) / 256
output = cv2.filter2D(img, -1, kernel)

cv2.imshow('Gaussian blur 5x5', output)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
