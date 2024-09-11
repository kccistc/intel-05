# #%%
# import matplotlib.pyplot as plt
# import numpy as np
# data_tuple = ("A", "B")
# data_list = list(data_tuple)

# data_list.append("C")

# data_tuple = tuple(data_list)

# print(data_tuple)
# # %%
# import matplotlib.pyplot as plt
# import numpy as np
# num = int(input("숫자 입력 :"))

# arr = np.arange(num*num)

# k = np.reshape(arr,(num,num))

# print(k)
# # %%
# import matplotlib.pyplot as plt
# import numpy as np
# num = int(input("숫자 입력 :"))

# arr = np.arange(num*num)

# k = np.reshape(arr,(1,num*num))

# print(k)
# %%
# import matplotlib.pyplot as plt
# import numpy as np
# import cv2
# cv2.__version__ 

# img = cv2.imread("/home/taejoon/workspace/practice/Lena.bmp", cv2.IMREAD_COLOR)
# img_expand = np.expand_dims(img,axis = 0)

# img_transpose = img_expand.transpose((0,3,2,1))


# %%
import matplotlib.pyplot as plt
import numpy as np
import cv2

    
img = cv2.imread("/home/taejoon/workspace/practice/Lena.bmp",cv2.IMREAD_GRAYSCALE)
kernel = np.array([ [1,1,1], [1,-8,1], [1,1,1]] )
print(kernel)

height, width = img.shape
temp_matrix = np.zeros((3,3))
temp_img = img.copy()

for i in range(0,height):
    for j in range(0,width):
        for k in range(-1,2):
            for m in range(-1,2):
                if (j+k <0 or j+k >= width or i+m < 0 or i+m >= height):
                    continue
                else:
                    temp_matrix[k+1,m+1] = img[j+k,i+m]
                
                
        temp_img[j,i]=(np.sum(temp_matrix*kernel))
        temp_matrix = np.zeros((3,3))

cv2.imshow('edge', temp_img)
cv2.waitKey(0)