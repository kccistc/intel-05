import cv2
import numpy as np

a = cv2.imread("./image/cat.jpg")

# 이미지가 정상적으로 읽혔는지 확인
if a is not None:
    # 이미지 창에 띄우기
    cv2.imshow("Cat Image", a)

    # 키 입력 대기 (0은 무한 대기)
    cv2.waitKey(0)

    # 모든 창 닫기
    cv2.destroyAllWindows()
else:
    print("이미지를 찾을 수 없습니다.")
    
b = np.expand_dims(a, 0)
print(b.shape)
c = np.transpose(b, (0,3,2,1))
print(c.shape)
