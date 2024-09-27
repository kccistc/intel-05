from ultralytics import YOLO
import cv2
import numpy as np
import requests
from paddleocr import PaddleOCR

# 네이버 API 정보 설정 (발급받은 Client ID, Client Secret 사용)
NAVER_CLIENT_ID = 'YOUR_CLIENT_ID'
NAVER_CLIENT_SECRET = 'YOUR_CLIENT_SECRET'

# YOLOv8 모델 로드
model = YOLO('yolov8x.pt')  # 성능이 더 높은 YOLOv8x 모델 사용

# PaddleOCR 초기화 (한글 + 영어 지원)
ocr_model = PaddleOCR(use_angle_cls=True, lang='korean')

# 전역 변수로 클릭된 좌표 저장
clicked_point = None
processing_done = False  # 객체 처리 완료 여부를 저장

# 마우스 클릭 이벤트 처리 함수
def mouse_callback(event, x, y, flags, param):
    global clicked_point, processing_done
    if event == cv2.EVENT_LBUTTONDOWN:  # 마우스 왼쪽 클릭
        clicked_point = (x, y)
        processing_done = False  # 클릭 시 다시 객체 탐지 시작
        print(f"Mouse clicked at: {x}, {y}")

# 이미지 전처리 함수 (노이즈 제거 및 텍스트 강조)
def preprocess_image(image):
    """OCR 인식률을 높이기 위한 이미지 전처리"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 그레이스케일로 변환
    denoised = cv2.fastNlMeansDenoising(gray, h=30)  # 노이즈 제거
    _, binary_image = cv2.threshold(denoised, 150, 255, cv2.THRESH_BINARY)  # 이진화
    
    # 추가 개선: 모폴로지 연산으로 텍스트를 더 선명하게
    kernel = np.ones((2, 2), np.uint8)
    processed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    
    return processed_image

# 세로 텍스트 이미지 회전 함수
def rotate_image_for_vertical_text(image):
    """세로 텍스트가 포함된 이미지를 90도 회전"""
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

# OCR 함수 정의 (PaddleOCR 이용)
def perform_ocr_with_paddleocr(image):
    """PaddleOCR를 사용하여 텍스트 인식"""
    # preprocessed_image = preprocess_image(image)  # 이미지 전처리
    
    # OCR 실행, 회전 검출을 위한 use_angle_cls 설정
    result = ocr_model.ocr(image, cls=True)

    # 결과 검증
    if result is None or len(result) == 0 or result[0] is None or len(result[0]) == 0:
        print("OCR failed: No text detected")
        return ""

    # 텍스트 추출 (글자 순서 유지를 위해 line[1][0]을 정렬)
    text_lines = [line[1][0] for line in sorted(result[0], key=lambda x: x[0][0][1])]
    
    detected_text = ' '.join(text_lines)
    return detected_text.strip()

# 네이버 책 검색 API 호출 함수
def search_book_info(book_title):
    """네이버 책 검색 API를 사용해 책 정보를 검색"""
    url = "https://openapi.naver.com/v1/search/book.json"
    headers = {
        "X-Naver-Client-Id": 'Gvr2kF63qLurl47g5B0w',
        "X-Naver-Client-Secret": 'XYesCCBa4V',
    }
    params = {"query": book_title, "display": 1}  # 책 제목으로 검색, 1개의 결과만 반환

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        result = response.json()
        if result['items']:
            # 첫 번째 검색 결과 반환
            return result['items'][0]
        else:
            return None
    else:
        print(f"Error: {response.status_code}")
        return None

# 책 정보를 화면에 표시하는 함수
def show_book_info(frame, book_info, x1, y1):
    """화면에 책 정보를 표시하는 함수"""
    # 책 정보 출력
    info_text = f"Title: {book_info['title']}\nAuthor: {book_info['author']}\nPublisher: {book_info['publisher']}"
    
    # 책 정보를 화면에 표시
    cv2.putText(frame, "Detected Book Info:", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, book_info['title'], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"Author: {book_info['author']}", (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"Publisher: {book_info['publisher']}", (x1, y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return frame

# 객체 감지 함수 정의 (클릭한 부분을 기준으로 책을 감지)
def detect_book_near_click(frame):
    """YOLOv8로 책을 감지하고 클릭한 좌표 근처의 책을 반환하는 함수"""
    global clicked_point
    results = model(frame)  # YOLOv8로 객체 감지

    # 결과에서 책 객체 필터링 (YOLOv8은 boxes.xyxy 형태로 결과 반환)
    boxes = results[0].boxes.xyxy.cpu().numpy()  # 박스 좌표
    confidences = results[0].boxes.conf.cpu().numpy()  # 신뢰도
    class_ids = results[0].boxes.cls.cpu().numpy()  # 클래스 ID

    closest_box = None
    closest_distance = float('inf')

    if clicked_point is not None:
        cx, cy = clicked_point  # 클릭한 좌표

        # 'book' 클래스 ID와 클릭한 좌표와의 거리를 기반으로 가장 가까운 책 선택
        for i, class_id in enumerate(class_ids):
            if model.names[int(class_id)] == 'book':
                x1, y1, x2, y2 = boxes[i]
                # 박스 중심 좌표 계산
                box_center_x = (x1 + x2) / 2
                box_center_y = (y1 + y2) / 2
                # 클릭한 좌표와의 거리 계산
                distance = np.sqrt((box_center_x - cx) ** 2 + (box_center_y - cy) ** 2)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_box = (int(x1), int(y1), int(x2), int(y2))

    clicked_point = None  # 클릭 처리 후 초기화

    return [closest_box] if closest_box else []

# 결과 그리기 및 네이버 API 호출 후 정보 표시 함수
def draw_boxes_and_ocr(frame):
    """감지된 한 권의 책 객체에 바운딩 박스와 텍스트를 그린 후 네이버 API로 책 정보를 검색하는 함수"""
    boxes = detect_book_near_click(frame)

    if len(boxes) > 0 and boxes[0] is not None:
        (x1, y1, x2, y2) = boxes[0]
        # 바운딩 박스 그리기
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # OCR 수행 (PaddleOCR 사용)
        book_roi = frame[y1:y2, x1:x2]  # 책의 옆면 영역
        title = perform_ocr_with_paddleocr(book_roi)

        # 책 제목 출력 및 네이버 API로 책 정보 검색
        if title:
            print(f"Detected book title: {title}")
            cv2.putText(frame, title, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # 네이버 API로 책 정보 검색
            book_info = search_book_info(title)
            if book_info:
                # 책 정보 출력 및 화면에 표시
                frame = show_book_info(frame, book_info, x1, y1)

    return frame

# 핸드폰 DroidCam 사용 설정 (웹캠 대체)
def run_droidcam():
    """DroidCam을 이용하여 핸드폰 카메라 비디오 스트림을 컴퓨터 화면에 출력"""
    # DroidCam이 웹캠처럼 인식되므로 기본 웹캠 인덱스(0) 또는 1번을 사용
    cap = cv2.VideoCapture(0)  # 0번 또는 1번으로 설정해야 할 수도 있음 (웹캠 번호 확인 필요)

    if not cap.isOpened():
        print("DroidCam 비디오 스트림에 연결할 수 없습니다.")
        return

    cv2.namedWindow("Book Detection, OCR & Search")
    cv2.setMouseCallback("Book Detection, OCR & Search", mouse_callback)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("비디오 종료")
            break

        global processing_done
        if not processing_done:
            # 결과 프레임에 그리기 및 OCR 수행 및 네이버 API 검색
            frame = draw_boxes_and_ocr(frame)
            processing_done = True

        # 결과 이미지 보여주기
        cv2.imshow('Book Detection, OCR & Search', frame)

        # 마우스 클릭이 발생할 때까지 대기
        while processing_done:  # 새로운 클릭이 발생할 때까지 대기
            cv2.waitKey(1)  # 1ms마다 대기하면서 루프 유지

        if cv2.waitKey(1) & 0xFF == 27:  # ESC 키를 누르면 종료
            break

    cap.release()
    cv2.destroyAllWindows()

# DroidCam을 사용하여 핸드폰 카메라로 객체 탐지 및 OCR 실행
run_droidcam()
