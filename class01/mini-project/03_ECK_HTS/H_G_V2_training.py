import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# CSV 파일 불러오기
train_df = pd.read_csv('/home/eckim/workspace/Project/Hand/Drone_hand_gesture.v2i.multiclass/train/_classes.csv')
valid_df = pd.read_csv('/home/eckim/workspace/Project/Hand/Drone_hand_gesture.v2i.multiclass/valid/_classes.csv')

# 열 이름 공백 제거
train_df.columns = train_df.columns.str.strip()
valid_df.columns = valid_df.columns.str.strip()


# 2. 이미지 데이터 제너레이터 설정
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

# 3. 이미지 데이터 로드
train_generator = train_datagen.flow_from_dataframe(
    train_df,
    directory='/home/eckim/workspace/Project/Hand/Drone_hand_gesture.v2i.multiclass/train',  # Train 이미지 파일이 저장된 디렉토리 경로
    x_col='filename',
    y_col=['Ascending', 'Descending', 'Pitch_Backward', 'Pitch_Forward', 'Roll_Left', 'Roll_Right', 'Yaw_Left', 'Yaw_Right'],  # 클래스 레이블
    target_size=(224, 224),
    batch_size=32,
    class_mode='raw'  # 다중 레이블 분류이므로 'raw'로 설정
)

valid_generator = valid_datagen.flow_from_dataframe(
    valid_df,
    directory='/home/eckim/workspace/Project/Hand/Drone_hand_gesture.v2i.multiclass/valid',  # Validation 이미지가 저장된 디렉토리 경로
    x_col='filename',
    y_col=['Ascending', 'Descending', 'Pitch_Backward', 'Pitch_Forward', 'Roll_Left', 'Roll_Right', 'Yaw_Left', 'Yaw_Right'],
    target_size=(224, 224),
    batch_size=32,
    class_mode='raw'
)

# 4. MobileNet 모델 정의
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 5. 커스텀 헤드 추가
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(8, activation='sigmoid')  # 8개의 클래스, 다중 레이블이므로 sigmoid 사용
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 6. 모델 학습
history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=10
)

# 학습이 완료되면 모델을 저장해두자
model.save('Hand_32_10_sigmoid.h5')