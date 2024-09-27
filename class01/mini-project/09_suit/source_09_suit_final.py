import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from openvino.runtime import Core
from torchvision.models import resnet50
import time
from scipy.spatial.distance import cosine  # 코사인 유사도 계산을 위해 사용

# OpenVINO Inference 모델 로드
ie = Core()
model = ie.read_model(model="./intel/person-detection-0202/FP32/person-detection-0202.xml", 
                      weights="./intel/person-detection-0202/FP32/person-detection-0202.bin")
compiled_model = ie.compile_model(model=model, device_name="CPU")
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# ResNet-50 모델 로드 (특징 추출용)
resnet_model = resnet50(pretrained=True)
resnet_model.eval()  # 모델을 평가 모드로 설정

# 이미지 전처리 (ResNet 입력 크기 맞추기)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 유사도 비교 임계값 설정 (값이 0에 가까울수록 더 유사)
similarity_threshold = 0.8789  # 이 값을 조정하여 비교 민감도를 설정할 수 있습니다

# 카메라 연결
cap = cv2.VideoCapture(0)  # 웹캠 사용

# 첫 번째 사람 인식 여부와 특징 벡터
first_recognized = False
known_features_list = []  # 첫 번째 사람의 특징 벡터를 저장할 리스트

# 캡처와 관련된 시간 변수 초기화
last_capture_time = 0  # 마지막으로 사진을 캡처한 시간
person_lost_time = 0  # 사람이 화면에서 사라진 시간
reset_threshold = 5  # 5초 동안 사람이 사라지면 나간 것으로 간주

# 프레임 카운터 초기화
frame_count = 0
start_time_total = time.time()  # 전체 처리 시간 측정을 위한 변수

def extract_features(frame):
    """ ResNet-50을 사용하여 주어진 프레임에서 특징 벡터를 추출하는 함수 """
    frame = transform(frame)  # 이미지 전처리
    frame = frame.unsqueeze(0)  # 배치 차원 추가 (1, C, H, W)
    with torch.no_grad():  # 그래디언트 계산 비활성화
        features = resnet_model(frame)
    return features.view(-1).numpy()  # 1D 벡터로 변환하여 반환

while cap.isOpened():
    start_time = time.time()  # 프레임 처리 시작 시간
    ret, frame = cap.read()
    if not ret:
        break

    # OpenVINO 모델로 사람 감지 수행
    input_image = cv2.resize(frame, (512, 512))  # 입력 이미지 크기 조정
    input_image = np.transpose(input_image, (2, 0, 1))  # (채널, 높이, 너비) 순서로 변환
    input_image = np.expand_dims(input_image, axis=0)  # 배치 차원 추가
    
    inference_start_time = time.time()  # 인퍼런스 시작 시간
    results = compiled_model([input_image])[output_layer]  # 감지 결과
    inference_time = (time.time() - inference_start_time) * 1000  # 인퍼런스 시간 (ms)

    person_detected = False  # 현재 프레임에서 사람 인식 여부
    other_person_detected = False  # 첫 번째 사람이 아닌 사람 감지 여부

    if results.ndim > 2:  # 결과가 다차원 배열일 경우
        for result in results[0][0]:
            if result[2] > 0.7:  # 감지된 객체의 신뢰도 (0.7 이상일 때)
                person_detected = True  # 사람 인식됨
                xmin = int(result[3] * frame.shape[1])
                ymin = int(result[4] * frame.shape[0])
                xmax = int(result[5] * frame.shape[1])
                ymax = int(result[6] * frame.shape[0])
                person_frame = frame[ymin:ymax, xmin:xmax]  # 감지된 사람 영역 추출

                # ResNet-50 모델을 사용하여 특징 벡터 추출
                extracted_features = extract_features(person_frame)

                if not first_recognized:
                    # 첫 번째 사람의 특징 저장
                    known_features_list.append(extracted_features)  # 특징 벡터를 리스트에 추가
                    first_recognized = True  # 첫 번째 인식 완료
                    print("처음 인식한 사람의 특징을 저장했습니다.")
                    
                    # 첫 번째 사람에게 초록색 바운딩 박스
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # 초록색 바운딩 박스
                    cv2.putText(frame, "USER", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                else:
                    # 기존의 첫 번째 사람 특징과 새로 인식된 사람 비교
                    # 여러 특징 벡터의 평균을 계산
                    avg_features = np.mean(known_features_list, axis=0)
                    similarity = 1 - cosine(avg_features, extracted_features)
                    print(f"유사도: {similarity:.4f}")

                    if similarity > similarity_threshold:
                        # 첫 번째 사람일 경우 초록색 바운딩 박스
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # 초록색 바운딩 박스
                        cv2.putText(frame, "USER", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        print("사용자입니다.")
                    else:
                        # 다른 사람일 경우 빨간색 바운딩 박스
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)  # 빨간색 바운딩 박스
                        cv2.putText(frame, "Other Person", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        print("다른 사람입니다.")
                        other_person_detected = True  # 다른 사람이 감지됨

                        # 새로운 사람의 사진을 10초마다 캡처
                        current_time = time.time()
                        if current_time - last_capture_time >= 10 and not person_detected and other_person_detected:
                            cv2.imwrite(f'captured_{int(current_time)}.jpg', frame)
                            last_capture_time = current_time  # 마지막 캡처 시간 업데이트
                            print("새로운 이미지가 저장되었습니다.")

    # 사람이 감지되었는지 여부에 따른 처리
    if person_detected:
        person_present = True  # 현재 사람이 있다고 표시
        person_lost_time = time.time()  # 사람이 감지된 시간을 갱신
    else:
        # 사람이 사라진 경우 5초 후에 리셋
        if time.time() - person_lost_time > reset_threshold:
            print("사람이 화면에서 사라졌습니다.")
            person_present = False  # 사람이 없는 상태로 전환

    # 전체 프레임 카운트 업데이트
    frame_count += 1
    total_time = time.time() - start_time_total  # 전체 시간
    fps = frame_count / total_time if total_time > 0 else 0  # FPS 계산

    # 결과 영상에 FPS와 인퍼런스 시간 출력
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)  # 텍스트 색상을 검은색으로 변경
    cv2.putText(frame, f"Inference Time: {inference_time:.2f} ms", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)  # 텍스트 색상을 검은색으로 변경

    # 결과 영상 출력
    cv2.imshow('Person Detection and Feature Extraction', frame)

    # 'q'를 눌러 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
