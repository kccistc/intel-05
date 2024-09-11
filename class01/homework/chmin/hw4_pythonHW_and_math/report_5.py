import cv2
import numpy as np

# 이미지 불러오기 (흑백)
img = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)

# 커널 정의
knl = np.array([[1, 1, 1],
                [1, -8, 1],
                [1, 1, 1]])
print(knl)

# custom_filter2D 함수 정의
def custom_filter2D(image, kernel):
    # 이미지와 커널의 크기 가져오기
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    # 패딩 크기 계산 (커널 크기를 기준으로 중앙에 맞게)
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    
    # 이미지에 제로 패딩 추가 (2D 이미지이므로 패딩도 2D로)
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
    
    # 출력 이미지를 저장할 배열 생성 (원본 이미지와 동일한 크기)
    output_image = np.zeros_like(image)
    
    # 필터 적용 (2D 컨볼루션)
    for i in range(image_height):
        for j in range(image_width):
            # 슬라이싱된 윈도우 가져오기
            region = padded_image[i:i+kernel_height, j:j+kernel_width]
            
            # 슬라이싱된 윈도우와 커널을 곱한 후 합산하여 출력 이미지에 저장
            output_image[i, j] = np.sum(region * kernel)
    
    return output_image

# custom_filter2D 적용
output = cv2.filter2D(img, -1, knl)

# 결과 출력
cv2.imshow('edge', output)
cv2.waitKey(0)
cv2.destroyAllWindows()
