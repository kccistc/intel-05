import json
import os
import cv2
import random
from tqdm import tqdm

# 데이터 경로 설정
data_root_path = '/home/jsh/workspace/paddleocr/dataset/'
save_root_path = '/home/jsh/workspace/paddleocr/dataset/preprocessed/'

# JSON 파일 로드
with open(os.path.join(data_root_path, 'textinthewild_data_info.json'), 'r', encoding='utf-8') as f:
    data = json.load(f)

# 책(book) 이미지만 필터링
book_images = [img for img in data['images'] if img['type'] == 'book']

# 데이터 분할 (train:val:test = 70:15:15)
random.shuffle(book_images)
n_train = int(len(book_images) * 0.7)
n_val = int(len(book_images) * 0.15)

train_images = book_images[:n_train]
val_images = book_images[n_train:n_train+n_val]
test_images = book_images[n_train+n_val:]

def process_dataset(images, split):
    os.makedirs(os.path.join(save_root_path, split, 'images'), exist_ok=True)
    label_file = open(os.path.join(save_root_path, f'{split}_label.txt'), 'w', encoding='utf-8')

    for image in tqdm(images, desc=f'Processing {split} set'):
        image_path = os.path.join(data_root_path, 'images', image['file_name'])
        img = cv2.imread(image_path)
        
        if img is None:
            continue

        # 이미지에 해당하는 annotation 찾기
        annotations = [ann for ann in data['annotations'] if ann['image_id'] == image['id'] and ann['attributes']['class'] == 'word']

        for idx, ann in enumerate(annotations):
            x, y, w, h = ann['bbox']
            
            # 잘못된 bbox 값 필터링
            if x <= 0 or y <= 0 or w <= 0 or h <= 0:
                continue

            text = ann['text']
            
            # 단어 영역 자르기
            crop_img = img[y:y+h, x:x+w]
            
            # 새 파일 이름 생성
            new_filename = f"{image['file_name'][:-4]}_{idx:03d}.jpg"
            save_path = os.path.join(save_root_path, split, 'images', new_filename)
            
            # 이미지 저장
            cv2.imwrite(save_path, crop_img)
            
            # 라벨 파일에 정보 추가 (PaddleOCR 형식)
            label_file.write(f"{split}/images/{new_filename}\t{text}\n")

    label_file.close()

# 데이터셋 처리
process_dataset(train_images, 'train')
process_dataset(val_images, 'val')
process_dataset(test_images, 'test')

print("데이터 전처리가 완료되었습니다.")