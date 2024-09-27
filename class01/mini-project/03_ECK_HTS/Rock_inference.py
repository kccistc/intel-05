import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image

# 저장된 모델 불러오기
model = tf.keras.models.load_model('mobilenet_batu_gunting_kertas.h5')

# 클래스 레이블 정의 (폴더 구조에 맞게 순서대로 들어가 있음)
class_labels = ['Rock', 'Scissors', 'Paper']

# 이미지 전처리 함수
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # MobileNetV2 모델 입력 크기로 맞춤
    img_array = image.img_to_array(img)  # 이미지를 배열로 변환
    img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가
    img_array = img_array / 255.0  # 0~1 사이로 정규화
    return img_array

# 이미지 파일 경로 설정
img_path = 'Kertas.jpg'  # 테스트할 이미지 경로를 여기에 입력

# 이미지 전처리
preprocessed_img = preprocess_image(img_path)

# 모델 추론
predictions = model.predict(preprocessed_img)

# softmax 결과에서 가장 큰 확률을 가진 클래스 인덱스 추출
predicted_class_index = np.argmax(predictions[0])
predicted_class_label = class_labels[predicted_class_index]

# 예측된 클래스 출력
print(f"Predicted Class: {predicted_class_label}")
print(f"Class Probabilities: {predictions[0]}")