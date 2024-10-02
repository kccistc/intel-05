import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time

model_path = '/home/han/Test/gesture_recognizer.task'

# MediaPipe Gesture Recognizer 설정
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# 결과를 저장할 전역 변수
recognized_gesture = 'No Gesture'
gesture_timer = 0  # 타이머 변수
gesture_detected = False  # 제스처 감지 여부 변수
start_time = 0  # 주먹 제스처 시작 시간을 저장할 변수






# 결과 콜백 함수
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global recognized_gesture, gesture_timer, gesture_detected, start_time
    if result.gestures and len(result.gestures) > 0:
        recognized_gesture = result.gestures[0][0].category_name
        
        # 주먹 제스처 감지
        if recognized_gesture == 'Closed_Fist':  # 'Fist'가 주먹 제스처의 카테고리 이름이라고 가정
            if not gesture_detected:
                start_time = time.time()  # 주먹 감지 시작 시간 저장
            gesture_detected = True
        else:
            gesture_detected = False
            gesture_timer = 0  # 다른 제스처가 인식되면 타이머 초기화
    else:
        recognized_gesture = 'No Gesture'
        gesture_detected = False
        gesture_timer = 0  # 아무 제스처도 인식되지 않으면 타이머 초기화

# Gesture Recognizer 옵션 설정
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result
)

# Gesture Recognizer 생성
with GestureRecognizer.create_from_options(options) as recognizer:
    # 웹캠 시작
    cap = cv2.VideoCapture(0)
    
    # 비디오의 너비와 높이를 설정
    w = 640
    h = 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    # 비디오 코덱 및 출력 파일 설정
    # 코덱은 플랫폼에 따라 다를 수 있습니다.
    # * 'XVID'는 Windows에서 자주 사용되며,
    # * 'mp4v'는 macOS나 Linux에서 자주 사용됩니다.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 코덱 설정
    out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (w, h))  # 30 FPS로 비디오 저장

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # OpenCV 이미지를 MediaPipe 이미지로 변환
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # 현재 프레임의 타임스탬프(ms)
        frame_timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)

        # 제스처 인식 수행
        recognizer.recognize_async(mp_image, frame_timestamp_ms)

        # 인식된 제스처를 웹캠 화면에 표시
        cv2.putText(frame, recognized_gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # 비디오 파일에 현재 프레임 저장
        out.write(frame)
        
        # 결과를 화면에 표시
        cv2.imshow('Webcam', frame)

        # 주먹이 감지되면 타이머를 체크
        if gesture_detected:
            elapsed_time = time.time() - start_time  # 경과 시간 계산
            if elapsed_time >= 3:  # 주먹이 3초 이상 감지되면
                print("Fist detected for 3 seconds. Exiting...")
                break
        else:
            start_time = 0  # 주먹이 감지되지 않으면 시작 시간 초기화

        if cv2.waitKey(5) & 0xFF == 27:  # ESC 키로 종료
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
