import cv2
import numpy as np

img = cv2.imread('image.png', cv2.IMREAD_COLOR)

identity_kernel = np.array([[0, 0, 0],[0, 1, 0], [0, 0, 0]])
ridge_kernel = np.array([[0, -1, 0],[-1, 4, -1], [0, -1, 0]])
edge_dectection_kernel = np.array([[-1, -1, -1],[-1, 8, -1], [-1, -1, -1]])
sharpen_kernel = np.array([[0, -1, 0],[-1, 5, -1], [0, -1, 0]])
box_blur_kernel = 1/9 * np.array([[1, 1, 1],[1, 1, 1], [1, 1, 1]])
gaussian_blur3_kernel = 1/16 * np.array([[1, 2, 1],[2, 4, 2], [1, 2, 1]])
gaussian_blur5_kernel = 1/256 * np.array([[1, 4, 6, 4, 1],[4, 16, 24, 16, 4], [6, 24, 36, 24, 6],[4, 16, 24, 16, 4],[1, 4, 6, 4, 1]])

output_identity = cv2.filter2D(img, -1, identity_kernel)
output_ridge = cv2.filter2D(img, -1, ridge_kernel)
output_edge_dectection = cv2.filter2D(img, -1, edge_dectection_kernel)
output_sharpen = cv2.filter2D(img, -1, sharpen_kernel)
output_box_blur = cv2.filter2D(img, -1, box_blur_kernel)
output_gaussian_blur3 = cv2.filter2D(img, -1, gaussian_blur3_kernel)
output_gaussian_blur5 = cv2.filter2D(img, -1, gaussian_blur5_kernel)

row1 = np.hstack((output_identity, output_ridge, output_edge_dectection))
row2 = np.hstack((output_sharpen, output_box_blur, output_gaussian_blur3))
row3 = np.hstack((output_gaussian_blur5, output_gaussian_blur5, output_gaussian_blur5)) 

combined_image = np.vstack((row1, row2, row3))

cv2.imshow('Combined Image', combined_image)

cv2.waitKey(0)
cv2.destroyAllWindows()