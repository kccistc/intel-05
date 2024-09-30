from transformers import BertTokenizer, BertForSequenceClassification
import torch
from ultralytics import YOLO
import cv2
import numpy as np
import requests
from paddleocr import PaddleOCR
from PIL import ImageFont, ImageDraw, Image
import urllib.request
from fuzzywuzzy import process


NAVER_CLIENT_ID = 
NAVER_CLIENT_SECRET = 


model = YOLO('yolov8x-seg.pt')
ocr_model = PaddleOCR(use_angle_cls=True, lang='korean', rec_model_dir='path/to/korean_server_v2.0_rec', use_gpu=True)


bert_model_name = 'monologg/kober0t'
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = BertForSequenceClassification.from_pretrained(bert_model_name)


with open('title_list.txt', 'r', encoding='utf-8') as f:
    correct_titles = [line.strip().strip('"') for line in f.readlines()]

clicked_point = None
processing_done = False 

def mouse_callback(event, x, y, flags, param):
    global clicked_point, processing_done
    if event == cv2.EVENT_LBUTTONDOWN:  
        clicked_point = (x, y)
        processing_done = False  
        print(f"Mouse clicked at: {x}, {y}")


def rotate_image_for_vertical_text(image):
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

def perform_ocr_with_paddleocr(image):
    image = rotate_image_for_vertical_text(image)
    if image is not None:
        cv2.imwrite("rotate.png", image)  
    result = ocr_model.ocr(image, cls=True)

    if result is None or len(result) == 0 or result[0] is None or len(result[0]) == 0:
        print("OCR failed: No text detected")
        return ""

    # x 좌표를 기준으로 정렬하여 글자 순서를 재정렬
    text_lines = [line[1][0] for line in sorted(result[0], key=lambda x: x[0][0][0])]
    detected_text = ' '.join(text_lines)
    return detected_text.strip()

def is_book_title_korean(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = bert_model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # 예: 첫 번째 클래스가 '책 제목 아님', 두 번째 클래스가 '책 제목'일 경우
    book_title_probability = predictions[0][1].item()  # 두 번째 클래스 확률
    return book_title_probability > 0.8  # 임계값 설정

def find_closest_title(ocr_text):
    closest_match = process.extractOne(ocr_text, correct_titles)
    if closest_match:
        return closest_match[0]  # 가장 유사한 제목 반환
    return None

def search_book_info(book_title):
    url = "https://openapi.naver.com/v1/search/book.json"
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
    }
    params = {"query": book_title, "display": 1}  # 책 제목으로 검색, 1개의 결과만 반환

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        result = response.json()
        if result['items']:
            return result['items'][0]
        else:
            return None
    else:
        print(f"Error: {response.status_code}")
        return None

def show_book_info_with_pil(frame, book_info, x1, y1):
    # 책 정보 텍스트 설정
    info_text = f"제목: {book_info['title']}\n저자: {book_info['author']}\n출판사: {book_info['publisher']}"

    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", 20)

    # 표지 이미지 가져오기
    cover_image_url = book_info.get('image', None)
    if cover_image_url:
        try:
            cover_image = Image.open(urllib.request.urlopen(cover_image_url))
            cover_image = cover_image.resize((100, 150))  # 이미지 크기 조정
            pil_img.paste(cover_image, (x1, y1 - 200))  # 표지 이미지를 책 상단에 표시
        except Exception as e:
            print(f"Error loading cover image: {e}")

    # 텍스트 표시
    draw.text((x1, y1 - 50), "책 정보:", font=font, fill=(255, 0, 0))
    draw.text((x1, y1), info_text, font=font, fill=(255, 0, 0))

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def detect_book_near_click(frame):
    global clicked_point
    results = model(frame)  
    masks = results[0].masks
    if masks is not None:
        masks = masks.data.cpu().numpy().astype(np.uint8)
        
        if masks.shape[1:] != frame.shape[:2]:
            masks = cv2.resize(masks, (frame.shape[1], frame.shape[0]))
            
        mask_3channel = np.repeat(masks[:,:,np.newaxis], 3, axis=2)
        
        frame = frame.astype(np.uint8)

        boxes = results[0].boxes.xyxy.cpu().numpy()  
        confidences = results[0].boxes.conf.cpu().numpy()  
        class_ids = results[0].boxes.cls.cpu().numpy()  

        closest_box = None
        closest_distance = float('inf')
        closet_distance_idx = 0
        if clicked_point is not None:
            cx, cy = clicked_point  
            for i, class_id in enumerate(class_ids):
                if model.names[int(class_id)] == 'book':
                    x1, y1, x2, y2 = boxes[i]
                    box_center_x = (x1 + x2) / 2
                    box_center_y = (y1 + y2) / 2
                    distance = np.sqrt((box_center_x - cx) ** 2 + (box_center_y - cy) ** 2)
                    if distance < closest_distance:
                        closest_distance = distance
                        closet_distance_idx = i
                        closest_box = (int(x1), int(y1), int(x2), int(y2))
     
        extract_result = cv2.bitwise_and(frame, frame, mask=masks[closet_distance_idx])
                   
        if extract_result is not None:
            cv2.imwrite("segmented_image.png", extract_result)  

        clicked_point = None  

        return [closest_box] if closest_box else [], extract_result

def draw_boxes_and_ocr(frame):
    boxes, extract_result = detect_book_near_click(frame)

    if len(boxes) > 0 and boxes[0] is not None:
        (x1, y1, x2, y2) = boxes[0]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        book_roi = extract_result[y1:y2, x1:x2]  
        title = perform_ocr_with_paddleocr(book_roi)

        # 제목 유사도 판단
        closest_title = find_closest_title(title)
        if closest_title:
            print(f"Detected book title: {closest_title}")
            
            book_info = search_book_info(closest_title)
            if book_info:
                frame = show_book_info_with_pil(frame, book_info, x1, y1)
        else:
            print("Detected text is not likely a book title.")

    return frame

def run_droidcam():
    cap = cv2.VideoCapture(0)  

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
            frame = draw_boxes_and_ocr(frame)
            processing_done = True

        cv2.imshow('Book Detection, OCR & Search', frame)

        while processing_done:
            cv2.waitKey(1)  

        if cv2.waitKey(1) & 0xFF == 27:  
            break

    cap.release()
    cv2.destroyAllWindows()

run_droidcam()
