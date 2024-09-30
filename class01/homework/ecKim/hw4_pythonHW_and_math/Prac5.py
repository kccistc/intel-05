import cv2
import numpy as np

img = cv2.imread('smile3.jpg',cv2.IMREAD_GRAYSCALE)
kernel=np.array([[1,1,1],[1,-8,1],[1,1,1]])
print(kernel)

x=img.shape[0]
y=img.shape[1]
print(x,y)
resized_img=np.zeros((x+2,y+2))
output_img=np.zeros((x,y))
output_img2=np.zeros((x,y))
print(resized_img)


for i in range(x):
    for j in range(y):
        resized_img[i+1,j+1]=img[i,j]


for i in range(x):
    for j in range(y):
        conValue=(resized_img[i,j]*kernel[0,0]\
        +resized_img[i+1,j]*kernel[0+1,0]\
        +resized_img[i+2,j]*kernel[0+2,0]\
        +resized_img[i+0,j+1]*kernel[0+0,1]\
        +resized_img[i+1,j+1]*kernel[0+1,1]\
        +resized_img[i+2,j+1]*kernel[0+2,1]\
        +resized_img[i+0,j+2]*kernel[0+0,2]\
        +resized_img[i+1,j+2]*kernel[0+1,2]\
        +resized_img[i+2,j+2]*kernel[0+2,2])/9
        output_img[i,j]=conValue

output_img2=cv2.filter2D(img,-1,kernel)
cv2.imshow('edge1',output_img)
cv2.imshow('edge2',output_img2)
cv2.waitKey(0)



img = cv2.imread('smile3.jpg', cv2.IMREAD_GRAYSCALE)

kernel = np.array([[1, 1, 1],
                   [1, -8, 1],
                   [1, 1, 1]])

x = img.shape[0]
y = img.shape[1]
print(x, y)

resized_img = np.zeros((x+2, y+2))
output_img = np.zeros((x, y))

resized_img[1:x+1, 1:y+1] = img

for i in range(x):
    for j in range(y):
        region = resized_img[i:i+3, j:j+3]
        conValue = np.sum(region * kernel)
        output_img[i, j] = conValue

output_img2 = cv2.filter2D(img, -1, kernel)

cv2.imshow('edge1', output_img)
cv2.imshow('edge2', output_img2)
cv2.waitKey(0)