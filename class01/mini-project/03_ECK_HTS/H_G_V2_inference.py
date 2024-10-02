import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# 저장된 모델 불러오기
model = tf.keras.models.load_model('Hand_32_10_sigmoid.h5')

# 클래스 레이블 정의
class_labels = ['Ascending', 'Descending', 'Pitch_Backward', 'Pitch_Forward', 'Roll_Left', 'Roll_Right', 'Yaw_Left', 'Yaw_Right']

# 이미지 전처리 함수
def preprocess_frame(frame):
    # OpenCV에서 가져온 이미지를 모델이 예측할 수 있는 형식으로 전처리
    img = cv2.resize(frame, (224, 224))  # 모델 입력 크기에 맞춰 리사이즈
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)  # 배치 차원을 추가
    img_array /= 255.0  # 0~1 사이로 정규화
    return img_array

# 웹캠에서 실시간으로 프레임 읽기
cap = cv2.VideoCapture(0)  # 0은 기본 웹캠을 의미함

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    # 이미지 전처리
    preprocessed_frame = preprocess_frame(frame)

    # 모델 추론
    predictions = model.predict(preprocessed_frame)

    # softmax 결과에서 가장 큰 확률을 가진 클래스 인덱스 추출
    predicted_class_index = np.argmax(predictions[0])  # 확률이 가장 큰 클래스의 인덱스
    predicted_class_label = class_labels[predicted_class_index]  # 해당 클래스 레이블

    # 예측된 클래스 출력
    output_text = f"Predicted: {predicted_class_label}"

    # 프레임에 텍스트 표시
    cv2.putText(frame, output_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # 화면에 표시
    cv2.imshow('Hand Gesture Recognition', frame)

    # 'q'를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠과 창 닫기
cap.release()
cv2.destroyAllWindows()