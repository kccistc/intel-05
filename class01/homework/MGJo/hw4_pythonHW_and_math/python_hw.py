#%%
# 튜플 a, b 생성
a = ('A','B')
print(a)

a = a + ('C', )  # c는 튜플 형태로 추가

print(a)


# %%
import numpy as np

n = int(input("정수를 입력하세요: "))

for i in range(n):
    for j in range(1 , n+1):
        print(i*n+j, end="\t")
    print()
# %%
import numpy as np

# 정수 n 입력받기
n = int(input("정수 n을 입력하세요: "))

# nxn 크기의 숫자 사각형 생성
matrix = np.arange(1, n*n + 1).reshape(n, n)

# 1차원 배열로 변환
array = matrix.reshape(-1)

# 1차원 배열 출력
print("1차원 배열로 변환된 결과:")
print(array)

# %%
import numpy as np
from PIL import Image


image_path = 'lena.jpg'  


image = Image.open(image_path)
image_array = np.array(image)


print("원본 이미지 배열의 shape:", image_array.shape)


expanded_array = np.expand_dims(image_array, axis=0)

print("차원이 확장된 이미지 배열의 shape:", expanded_array.shape)

transposed_array = np.transpose(expanded_array, (0, 3, 2, 1))

print("변경된 이미지 배열의 shape:", transposed_array.shape)

# %%
import cv2
import numpy as np

def convolution2d(image, kernel):
    
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    # 커널의 중심
    k_height, k_width = kernel_height // 2, kernel_width // 2
    
    output = np.zeros_like(image)
    
   
    for y in range(image_height):
        for x in range(image_width):
           
            if x - k_width >= 0 and x + k_width < image_width and y - k_height >= 0 and y + k_height < image_height:
               
                region = image[y - k_height:y + k_height + 1, x - k_width:x + k_width + 1]
               
                output[y, x] = np.sum(region * kernel)
    
    output = np.clip(output, 0, 255).astype(np.uint8)
    
    return output

# 이미지 읽기 (그레이스케일)
img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)

# 커널 정의
kernel = np.array([[1,1,1],[1, -8, 1], [1,1,1]])

# 직접 구현한 합성곱 연산
output = convolution2d(img, kernel)

# 결과 이미지 표시
cv2.imshow('edge', output)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
