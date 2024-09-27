import torch
import cv2
import numpy as np

# YOLOv5 모델 로드
model = torch.hub.load("ultralytics/yolov5", "yolov5s")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("카메라 탐지 불가")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 모델로 사람 탐지
    results = model(frame)
    detections = results.pred[0]  # 예측 결과 가져오기
    person_detections = detections[detections[:, 5] == 0]  # 사람 클래스만 필터링

    for det in person_detections:
        # Bounding box 좌표 추출 (x1, y1, x2, y2)
        x1, y1, x2, y2, conf, cls = det

        # 바운딩 박스 그리기
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # 신뢰도(confidence) 표시
        label = f"Person {conf:.2f}"
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        cropped_person = frame[int(y1):int(y2), int(x1):int(x2)]

    # 결과 프레임 출력ㅂ
    cv2.imshow('YOLOv5 - Person Detection', frame)

    key = cv2.waitKey(24)
    if key & 0xFF == ord('q'):  # 'q' 키를 눌러 종료
        break

cap.release()
cv2.destroyAllWindows()