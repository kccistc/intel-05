import cv2
import numpy as np
import matplotlib.pyplot as plt

img_1 = cv2.imread('./image/lama.png')
img = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)

kernel_identify = np.array([[0, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]])
kernel_edge1 = np.array([[0, -1, 0],
                         [-1, 4, -1],
                         [0, -1, 0]])
kernel_edge2 = np.array([[-1, -1, -1],
                         [-1, 8, -1],
                         [-1, -1, -1]])
kernel_sharpen = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
kernel_box_blur = np.array([[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]])/9
kernel_Gaussian3 = np.array([[1, 2, 1],
                             [2, 4, 2],
                             [1, 2, 1]])/16
kernel_Gaussian5 = np.array([[1, 4, 6, 4, 1],
                             [4, 16, 24, 16, 4],
                             [6, 24, 36, 24, 6],
                             [4, 16, 24, 16, 4],
                             [1, 4, 6, 4, 1]])/256

print(kernel_identify)
print(kernel_edge1)
print(kernel_edge2)
print(kernel_sharpen)
print(kernel_box_blur)
print(kernel_Gaussian3)
print(kernel_Gaussian5)

output1 = cv2.filter2D(img, -1, kernel_identify)
output2 = cv2.filter2D(img, -1, kernel_edge1)
output3 = cv2.filter2D(img, -1, kernel_edge2)
output4 = cv2.filter2D(img, -1, kernel_sharpen)
output5 = cv2.filter2D(img, -1, kernel_box_blur)
output6 = cv2.filter2D(img, -1, kernel_Gaussian3)
output7 = cv2.filter2D(img, -1, kernel_Gaussian5)

outputs = [ img, output1, output2, output3, output4, output5, output6, output7 ]
titles = [ 'riginal', 'Identify', 'Edge1', 'Edge2', 'Sharpen', 'Box Blur', 'Gaussian3', 'Gaussian5' ]

plt.figure(figsize=(15,10))

for i in range(8):
    plt.subplot(2,4,i+1)
    plt.imshow(outputs[i])
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()