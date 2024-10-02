
# word=('A','B')
# wlist=list(word)
# wlist.append("C")
# wTu=tuple(wlist)

# print(word)
# print(wTu)
#%%

# num=int(input("정수 n을 입력하세요 :"))

# a=1

# for i in range(num):
#     for j in range(num):
#         print(a,end=" ")
#         a+=1
#     print()
        
# %%
# import numpy as np

# a= np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
# b= a.reshape(-1)

# print(a)
# print(b)

#%%

# import cv2
# import numpy as np

# Img = cv2.imread('dog.png')
# ADD_img = np.expand_dims(Img,axis=0)

# Trans_img = ADD_img.transpose()

# print(ADD_img.shape)
# print(Trans_img.shape)

import cv2
import numpy as np

img = cv2.imread('dog.png', cv2.IMREAD_GRAYSCALE)

kernel = np.array([[1,1,1],[1,-8,1],[1,1,1]])

img_height, img_width = img.shape
kernel_height, kernel_width = kernel.shape

output = np.zeros((img_height, img_width), dtype=np.float32)

pad_height = kernel_height // 2
pad_width = kernel_width // 2
padded_img = np.pad(img, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

for i in range(pad_height, img_height + pad_height):
    for j in range(pad_width, img_width + pad_width):
        region = padded_img[i - pad_height:i + pad_height + 1, j - pad_width:j + pad_width + 1]
        output[i - pad_height, j - pad_width] = np.sum(region * kernel)

output = np.clip(output, 0, 255).astype(np.uint8)

cv2.imshow('edge', output)
cv2.waitKey(0)
cv2.destroyAllWindows()
