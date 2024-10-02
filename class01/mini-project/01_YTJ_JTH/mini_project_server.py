import base64
from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from paddleocr import PaddleOCR
from transformers import BertTokenizer, BertForSequenceClassification
from ultralytics import YOLO
from PIL import Image as PILImage, ImageFont, ImageDraw
import urllib.request
from fuzzywuzzy import process
import requests

app = Flask(__name__)

# 모델 로드
model = YOLO('yolov8x-seg.pt')
ocr_model = PaddleOCR(use_angle_cls=True, lang='korean', use_gpu=True)

# KoBERT 모델 및 토크나이저 로드bert_model
bert_model_name = 'monologg/kobert'
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = BertForSequenceClassification.from_pretrained(bert_model_name)

# 네이버 API 정보 설정
NAVER_CLIENT_ID = 'Gvr2kF63qLurl47g5B0w'
NAVER_CLIENT_SECRET = 'XYesCCBa4V'

# 정답 제목 목록 로드
with open('title_list.txt', 'r', encoding='utf-8') as f:
    correct_titles = [line.strip().strip('"') for line in f.readlines()]

def rotate_image_for_vertical_text(image):
    """세로 텍스트가 포함된 이미지를 90도 회전"""
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

def process_frame(frame, click_x, click_y):
    """YOLO로 책을 감지하고 OCR 및 KoBERT를 통해 책 제목을 추출한 후 네이버 API로 정보를 검색"""
    results = model(frame)
    masks = results[0].masks.data.cpu().numpy().astype(np.uint8)  # 세그멘테이션 마스크
    boxes = results[0].boxes.xyxy.cpu().numpy()  # 책 객체들의 경계박스 좌표 가져오기
    class_ids = results[0].boxes.cls.cpu().numpy()  # 감지된 객체들의 클래스 ID

    closest_box = None
    closest_mask = None
    closest_distance = float('inf')

    for i, class_id in enumerate(class_ids):
        if model.names[int(class_id)] == 'book':
            x1, y1, x2, y2 = boxes[i]
            box_center_x = (x1 + x2) / 2
            box_center_y = (y1 + y2) / 2
            distance = np.sqrt((box_center_x - click_x) ** 2 + (box_center_y - click_y) ** 2)
            if distance < closest_distance:
                closest_distance = distance
                closest_box = (int(x1), int(y1), int(x2), int(y2))
                closest_mask = masks[i]

    if closest_box and closest_mask is not None:
        x1, y1, x2, y2 = closest_box
        book_roi = frame[y1:y2, x1:x2]

        # 이미지 회전 적용
        book_roi = rotate_image_for_vertical_text(book_roi)

        # OCR 수행
        title = perform_ocr_with_paddleocr(book_roi)

        # 유사한 제목 찾기
        closest_title = find_closest_title(title)

        if closest_title:
            book_info = search_book_info(closest_title)

            # 세그멘테이션 마스크를 PNG 이미지로 인코딩
            mask_encoded = cv2.imencode('.png', closest_mask * 255)[1].tobytes()
            mask_base64 = base64.b64encode(mask_encoded).decode('utf-8')

            return closest_title, book_info, mask_base64, (x1, y1, x2, y2)

    return "No book detected.", None, None, None

def perform_ocr_with_paddleocr(image):
    """PaddleOCR를 사용하여 텍스트 인식"""
    result = ocr_model.ocr(image, cls=True)
    if result and len(result[0]) > 0:
        text_lines = [line[1][0] for line in sorted(result[0], key=lambda x: x[0][0][0])]
        return ' '.join(text_lines)
    return ""

def find_closest_title(ocr_text):
    """OCR 텍스트와 가장 유사한 제목을 찾는 함수"""
    closest_match = process.extractOne(ocr_text, correct_titles)
    return closest_match[0] if closest_match else None

def search_book_info(book_title):
    """네이버 책 검색 API를 사용해 책 정보를 검색"""
    url = "https://openapi.naver.com/v1/search/book.json"
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
    }
    params = {"query": book_title, "display": 1}
    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        result = response.json()
        if result['items']:
            return result['items'][0]
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame_request():
    image_data = request.files['image'].read()
    click_x = float(request.form['click_x'])
    click_y = float(request.form['click_y'])
    
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    title, book_info, mask, box = process_frame(img, click_x, click_y)

    if book_info and mask:
        return jsonify({
            'title': title,
            'author': book_info['author'],
            'publisher': book_info['publisher'],
            'image': book_info.get('image', None),
            'mask': mask
        })

    return jsonify({'error': 'Book not found or no information available'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
