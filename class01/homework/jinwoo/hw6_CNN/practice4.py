import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer

# CNN 모델 생성 및 학습
def create_cnn_model():
    model = Sequential([
        InputLayer(input_shape=(64, 64, 3)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # For binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ANN 모델 생성 및 학습
def create_ann_model():
    model = Sequential([
        InputLayer(input_shape=(64, 64, 3)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # For binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 데이터셋 준비 (학습 및 테스트 데이터 로드)
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)

training_set = train_datagen.flow_from_directory(
    './chest_xray/train',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

test_set = test_datagen.flow_from_directory(
    './chest_xray/test',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# CNN 모델 학습 및 저장
cnn_model = create_cnn_model()
cnn_model.fit(training_set, epochs=10, validation_data=test_set)
cnn_model.save('cnn_model.h5')

# ANN 모델 학습 및 저장
ann_model = create_ann_model()
ann_model.fit(training_set, epochs=10, validation_data=test_set)
ann_model.save('ann_model.h5')

# CNN 모델 로드
cnn_model = tf.keras.models.load_model('cnn_model.h5')
# ANN 모델 로드
ann_model = tf.keras.models.load_model('ann_model.h5')

# CNN 모델 평가
cnn_loss, cnn_acc = cnn_model.evaluate(test_set, steps=len(test_set))
print(f'CNN 모델 테스트 정확도: {cnn_acc * 100:.2f}%')

# ANN 모델 평가
ann_loss, ann_acc = ann_model.evaluate(test_set, steps=len(test_set))
print(f'ANN 모델 테스트 정확도: {ann_acc * 100:.2f}%')

# 성능 비교를 위한 시각화
labels = ['CNN', 'ANN']
accuracy = [cnn_acc, ann_acc]
loss = [cnn_loss, ann_loss]

plt.figure(figsize=(12, 5))

# 정확도 비교
plt.subplot(1, 2, 1)
plt.bar(labels, accuracy, color=['blue', 'green'])
plt.title('Accuracy Comparison')
plt.ylabel('Accuracy')

# 손실 비교
plt.subplot(1, 2, 2)
plt.bar(labels, loss, color=['blue', 'green'])
plt.title('Loss Comparison')
plt.ylabel('Loss')

plt.tight_layout()
plt.show()

# 테스트 세트 예측 및 혼동 행렬 생성
y_true = test_set.classes
y_pred_cnn = (cnn_model.predict(test_set) > 0.5).astype(int).flatten()
y_pred_ann = (ann_model.predict(test_set) > 0.5).astype(int).flatten()

print("\nCNN Classification Report:\n", classification_report(y_true, y_pred_cnn))
print("ANN Classification Report:\n", classification_report(y_true, y_pred_ann))

# 혼동 행렬 시각화
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

conf_matrix_cnn = confusion_matrix(y_true, y_pred_cnn)
conf_matrix_ann = confusion_matrix(y_true, y_pred_ann)

# CNN 혼동 행렬
ax[0].matshow(conf_matrix_cnn, cmap='coolwarm')
ax[0].set_title('CNN Confusion Matrix')
ax[0].set_xlabel('Predicted')
ax[0].set_ylabel('True')

# ANN 혼동 행렬
ax[1].matshow(conf_matrix_ann, cmap='coolwarm')
ax[1].set_title('ANN Confusion Matrix')
ax[1].set_xlabel('Predicted')
ax[1].set_ylabel('True')

plt.show()
