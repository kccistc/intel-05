import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from openvino.runtime import Core
from efficientnet_pytorch import EfficientNet
import time
from scipy.spatial.distance import cosine

# OpenVINO Inference 모델 로드
ie = Core()
model = ie.read_model(model="./intel/person-detection-0202/FP32/person-detection-0202.xml", 
                      weights="./intel/person-detection-0202/FP32/person-detection-0202.bin")
compiled_model = ie.compile_model(model=model, device_name="CPU")
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# EfficientNet-B0 모델 로드 (특징 추출용)
efficientnet_model = EfficientNet.from_pretrained('efficientnet-b0')
efficientnet_model.eval()  # 모델을 평가 모드로 설정

# 이미지 전처리 (EfficientNet 입력 크기 맞추기)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 유사도 비교 임계값 설정
similarity_threshold = 0.9  # 유사도 민감도 조정

# 메모리 관리 설정
known_features_list = []  # 첫 번째 사람의 특징 벡터 리스트
recent_weight = 0.7  # 최근 특징 벡터 가중치
max_features = 10  # 저장할 최대 특징 벡터 개수

# 캡처와 관련된 시간 변수 초기화
last_capture_time = 0  # 마지막으로 사진을 캡처한 시간
person_lost_time = 0  # 사람이 화면에서 사라진 시간
reset_threshold = 5  # 5초 동안 사람이 사라지면 나간 것으로 간주

# 프레임 카운터 초기화
frame_count = 0
start_time_total = time.time()  # 전체 처리 시간 측정을 위한 변수

# 사람이 감지된 상태
first_recognized = False  # 첫 번째 사람 인식 여부

# 비동기 추론을 위한 플래그
async_inference = False

def extract_features(frame):
    """ EfficientNet-B0을 사용하여 주어진 프레임에서 특징 벡터를 추출하는 함수 """
    frame = transform(frame)  # 이미지 전처리
    frame = frame.unsqueeze(0)  # 배치 차원 추가 (1, C, H, W)
    with torch.no_grad():
        features = efficientnet_model(frame)
    return features.view(-1).numpy()  # 1D 벡터로 변환하여 반환

def detect_person(frame):
    """ OpenVINO 모델을 사용해 프레임에서 사람을 감지하는 함수 """
    input_image = cv2.resize(frame, (512, 512))  # 입력 이미지 크기 조정
    input_image = np.transpose(input_image, (2, 0, 1))  # (채널, 높이, 너비) 순서로 변환
    input_image = np.expand_dims(input_image, axis=0)  # 배치 차원 추가
    return compiled_model([input_image])[output_layer]  # 비동기 추론 결과 반환

def compare_features(extracted_features, known_features_list):
    """ 기존 첫 번째 사람의 특징 벡터와 새로운 특징 벡터 간의 유사도를 계산하는 함수 """
    avg_features = recent_weight * extracted_features + (1 - recent_weight) * np.mean(known_features_list, axis=0)
    similarity = 1 - cosine(avg_features, extracted_features)
    return similarity

def draw_bounding_box(frame, xmin, ymin, xmax, ymax, is_user):
    """ 인식된 사람에 대해 바운딩 박스를 그리는 함수 """
    color = (0, 255, 0) if is_user else (0, 0, 255)  # 사용자라면 초록색, 아니면 빨간색
    label = "USER" if is_user else "Other Person"
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
    cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def display_status(frame, status):
    """ 화면에 현재 상태 메시지를 출력하는 함수 """
    cv2.putText(frame, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

def save_image(frame):
    """ 새로운 사람의 이미지를 캡처하고 저장하는 함수 """
    current_time = time.time()
    cv2.imwrite(f'captured_{int(current_time)}.jpg', frame)
    print("새로운 이미지가 저장되었습니다.")

# 카메라 연결
cap = cv2.VideoCapture(0)

while cap.isOpened():
    start_time = time.time()  # 프레임 처리 시작 시간
    ret, frame = cap.read()
    if not ret:
        print("카메라에서 프레임을 읽을 수 없습니다.")
        break

    # 비동기 추론
    results = detect_person(frame)
    
    person_detected = False  # 현재 프레임에서 사람 인식 여부
    other_person_detected = False  # 첫 번째 사람이 아닌 사람 감지 여부

    # 사람 감지 결과 처리
    if results.ndim > 2:
        for result in results[0][0]:
            if result[2] > 0.5:  # 신뢰도가 0.7 이상인 객체만
                person_detected = True
                xmin = int(result[3] * frame.shape[1])
                ymin = int(result[4] * frame.shape[0])
                xmax = int(result[5] * frame.shape[1])
                ymax = int(result[6] * frame.shape[0])
                person_frame = frame[ymin:ymax, xmin:xmax]  # 사람 영역 추출
                
                # EfficientNet-B0 특징 추출
                extracted_features = extract_features(person_frame)

                if not first_recognized:
                    # 첫 번째 사람의 특징 저장
                    known_features_list.append(extracted_features)
                    first_recognized = True
                    draw_bounding_box(frame, xmin, ymin, xmax, ymax, True)
                    display_status(frame, "처음 사람 인식됨")
                else:
                    # 유사도 비교
                    similarity = compare_features(extracted_features, known_features_list)
                    print(f"유사도: {similarity:.4f}")

                    if similarity > similarity_threshold:
                        draw_bounding_box(frame, xmin, ymin, xmax, ymax, True)
                        display_status(frame, "사용자")
                    else:
                        draw_bounding_box(frame, xmin, ymin, xmax, ymax, False)
                        display_status(frame, "다른 사람")
                        other_person_detected = True

                        # 10초마다 새로운 사람의 사진 저장
                        if time.time() - last_capture_time >= 10:
                            save_image(frame)
                            last_capture_time = time.time()

                # 메모리 관리: 최대 특징 벡터 개수 유지
                if len(known_features_list) > max_features:
                    known_features_list.pop(0)

    # 사람이 감지되지 않았을 때
    if not person_detected and time.time() - person_lost_time > reset_threshold:
        print("사람이 화면에서 사라졌습니다.")
        first_recognized = False  # 사람 인식 초기화

    # FPS 계산 및 표시
    frame_count += 1
    total_time = time.time() - start_time_total
    fps = frame_count / total_time if total_time > 0 else 0
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # 결과 영상 출력
    cv2.imshow('Person Detection and Feature Extraction', frame)

    # 'q'를 눌러 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

