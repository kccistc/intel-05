import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def run_pose_estimation_mediapipe(source=0):
    cap = cv2.VideoCapture(source)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty frame.")
            continue
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # MediaPipe Pose 모델을 통해 결과 추출
        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            # 어깨와 골반 좌표 추출 (오른쪽 어깨: 12, 왼쪽 어깨: 11, 오른쪽 골반: 24, 왼쪽 골반: 23)
            landmarks = results.pose_landmarks.landmark
            right_shoulder = landmarks[12]
            left_shoulder = landmarks[11]
            right_hip = landmarks[24]
            left_hip = landmarks[23]
            
            # 2D 좌표 변환 (0~1 범위 값을 이미지 좌표로 변환)
            image_height, image_width, _ = image.shape
            right_shoulder_coords = (int(right_shoulder.x * image_width), int(right_shoulder.y * image_height))
            left_shoulder_coords = (int(left_shoulder.x * image_width), int(left_shoulder.y * image_height))
            right_hip_coords = (int(right_hip.x * image_width), int(right_hip.y * image_height))
            left_hip_coords = (int(left_hip.x * image_width), int(left_hip.y * image_height))
            
            # 어깨 y 좌표 중 작은 값 선택
            shoulder_y_min = max(right_shoulder_coords[1], left_shoulder_coords[1])
            
            # 사각형의 왼쪽 위와 오른쪽 아래 좌표 설정
            top_left = (left_hip_coords[0], shoulder_y_min)
            bottom_right = (right_hip_coords[0], right_hip_coords[1])
            
            # 사각형 그리기
            cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 2)  # 파란색 사각형
            
            # 어깨와 골반의 키포인트 그리기
            cv2.circle(image, right_shoulder_coords, 5, (0, 255, 0), -1)  # 오른쪽 어깨
            cv2.circle(image, left_shoulder_coords, 5, (0, 255, 0), -1)  # 왼쪽 어깨
            cv2.circle(image, right_hip_coords, 5, (0, 255, 0), -1)  # 오른쪽 골반
            cv2.circle(image, left_hip_coords, 5, (0, 255, 0), -1)  # 왼쪽 골반
        
        # 영상 출력
        cv2.imshow('MediaPipe Pose Estimation', image)
        
        if cv2.waitKey(5) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

# 실행
run_pose_estimation_mediapipe()