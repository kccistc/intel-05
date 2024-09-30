import cv2
import numpy as np
img = cv2.imread('lama.png', cv2.COLOR_BGR2RGB)

kernels = {
    "kernel1": np.array([[1, 1, 1],[1, -8, 1], [1, 1, 1]]),
    "kernel2": np.array([[0,0,0],[0,1,0],[0,0,0]]),
    "kernel3": np.array([[0,-1,0],[-1,4,-1],[0,-1,0]]),
    "kernel4": np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]),
    "kernel5": np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]),
    "kernel6": (1/9) * np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
    "kernel7": (1/16) * np.array([[1,2,1],[2,4,2],[1,2,1]]),
    "kernel8": (1/256) * np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]])
}

outputs = []

for name, kernel in kernels.items():
    output = cv2.filter2D(img, -1, kernel)
    outputs.append(output)

Filtered_output = np.hstack(outputs)
cv2.imshow('Kernels Images', Filtered_output)

cv2.waitKey(0)
cv2.destroyAllWindows()