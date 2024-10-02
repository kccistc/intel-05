import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

# 1. 이미지 데이터 제너레이터 설정
train_datagen = ImageDataGenerator(
    rescale=1./255,              # 모든 픽셀을 0-1 사이로 정규화
    shear_range=0.2,             # 랜덤으로 이미지의 기울기 적용
    zoom_range=0.2,              # 랜덤으로 이미지 확대
    horizontal_flip=True,        # 랜덤으로 이미지를 좌우 반전
)

val_datagen = ImageDataGenerator(rescale=1./255)  # Validation data는 정규화만 진행

# 2. 이미지 데이터 로드
train_generator = train_datagen.flow_from_directory(
    '/home/eckim/workspace/Project/Hand/Rock/train',  # Train 이미지가 있는 경로 (ex. 'dataset/train')
    target_size=(224, 224),     # MobileNetV2의 입력 크기
    batch_size=32,
    class_mode='categorical',   # 다중 클래스 분류이므로 'categorical'
    shuffle=True                # 데이터 셔플링
)

val_generator = val_datagen.flow_from_directory(
    '/home/eckim/workspace/Project/Hand/Rock/val',    # Validation 이미지가 있는 경로 (ex. 'dataset/val')
    target_size=(224, 224),     # MobileNetV2의 입력 크기
    batch_size=32,
    class_mode='categorical',   # 다중 클래스 분류이므로 'categorical'
)

num_classes = len(train_generator.class_indices)
print(f"Number of classes: {num_classes}")

# 3. MobileNetV2 모델 정의
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 4. 커스텀 헤드 추가
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(3, activation='softmax')  # 3개의 클래스이므로 출력층은 3
])

# 5. 모델 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 6. 모델 학습
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10  # 에포크 수는 상황에 맞게 조정
)

# 7. 학습 완료 후 모델 저장
model.save('mobilenet_batu_gunting_kertas.h5')