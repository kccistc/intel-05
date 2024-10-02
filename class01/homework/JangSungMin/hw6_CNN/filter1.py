import cv2
import numpy as np

# 이미지 로드 (컬러로 변경)
img = cv2.imread('lama.png', cv2.IMREAD_COLOR)

# 다양한 필터 커널 정의
identity_kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
ridge_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
edge_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
box_kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9
gaussian_3x3_kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
gaussian_5x5_kernel = np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]]) / 256
unsharp_kernel = np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]]) / 256

# 필터 리스트
kernels = {
    "Identity": identity_kernel,
    "Ridge": ridge_kernel,
    "Edge Detection": edge_kernel,
    "Sharpen": sharpen_kernel,
    "Box Blur": box_kernel,
    "Gaussian 3x3": gaussian_3x3_kernel,
    "Gaussian 5x5": gaussian_5x5_kernel,
    "Unsharp Mask": unsharp_kernel
}

# 각 필터를 적용하고 이미지를 저장
filtered_images = []
for kernel_name, kernel in kernels.items():
    output = cv2.filter2D(img, -1, kernel)
    # 필터 결과에 텍스트 추가 (필터 이름)
    output = cv2.putText(output.copy(), kernel_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                         0.7, (255, 255, 255), 2, cv2.LINE_AA)
    filtered_images.append(output)

# 이미지를 4x2 배열로 결합 (필요시 크기 조정)
row1 = cv2.hconcat(filtered_images[:4])  # 첫 4개의 필터 결합
row2 = cv2.hconcat(filtered_images[4:])  # 나머지 4개의 필터 결합

# 두 행을 결합하여 최종 이미지 생성
final_image = cv2.vconcat([row1, row2])

# 최종 이미지 출력
cv2.imshow('All Kernels Applied', final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
