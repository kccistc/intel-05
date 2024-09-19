import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 데이터 경로 설정
train_dir = '/path/to/chest_xray/train'
val_dir = '/path/to/chest_xray/val'
test_dir = '/path/to/chest_xray/test'

# ImageDataGenerator를 사용하여 데이터를 전처리
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# 이미지 데이터를 불러와서 배치 단위로 생성
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=32, class_mode='binary')
val_generator = val_datagen.flow_from_directory(val_dir, target_size=(150, 150), batch_size=32, class_mode='binary')
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(150, 150), batch_size=32, class_mode='binary')

from tensorflow.keras import layers, models

def build_cnn_model():
    model = models.Sequential()

    # CNN 레이어 추가
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

cnn_model = build_cnn_model()
cnn_model.summary()

history = cnn_model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

def build_ann_model():
    model = models.Sequential()

    # ANN 모델에서는 이미지를 1D로 변환하여 입력
    model.add(layers.Flatten(input_shape=(150, 150, 3)))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

ann_model = build_ann_model()
ann_model.summary()

history_ann = ann_model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

import matplotlib.pyplot as plt

def plot_history(cnn_history, ann_history):
    # CNN 모델의 정확도 및 손실
    plt.plot(cnn_history.history['val_accuracy'], label='CNN Validation Accuracy')
    plt.plot(ann_history.history['val_accuracy'], label='ANN Validation Accuracy')
    
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.legend()
    plt.show()

plot_history(history, history_ann)

