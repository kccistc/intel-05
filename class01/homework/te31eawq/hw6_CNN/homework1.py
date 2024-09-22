# %%
import cv2
import numpy as np

# 이미지 읽기 (그레이스케일)
img = cv2.imread('lama.png', cv2.COLOR_BGR2RGB)

# 커널 정의
kernel = np.array([[1, 4, 6, 4, 1],
                   [4, 16, 24, 16, 4],
                   [6, 24, 36, 24, 6],
                   [4, 16, 24, 16, 4],
                   [1, 4, 6, 4, 1]], dtype=np.uint8) / 256

# 필터 적용
output = cv2.filter2D(img, -1, kernel)

# 필터 결과 정규화
output_clipped = np.clip(output, 0, 255).astype(np.uint8)

# 결과를 BGR로 변환 (그레이스케일에서 BGR로 변환)

# 결과 출력
cv2.imshow('Filtered Image', output_clipped)
cv2.waitKey(0)
cv2.destroyAllWindows()

