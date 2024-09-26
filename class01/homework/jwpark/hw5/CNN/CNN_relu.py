import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
import numpy as np
import pickle

#mnist = tf.keras.datasets.mnist
fashion_mnist = tf.keras.datasets.fashion_mnist

(image_t, label_t), (image_test, label_test)= fashion_mnist.load_data()
image_t, image_test = image_t/255.0, image_test/255.0

# 6만개의 데이터 셋 중 1000개만 추출
# image_train 데이터는 번호, width, height로 구성됨
image_train = image_t[:1000,:,:]
label_train = label_t[:1000]
image_val = image_t[1000:1200,:,:]
label_val = label_t[1000:1200]
val_dataset = (image_val, label_val)

#class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize=(10, 10))
for i in range(10):
    plt.subplot(3, 4, i+1)
    # 축 눈금 표시
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image_t[i])
    plt.xlabel(class_names[label_t[i]])
plt.show()

# CNN
model = Sequential()
model.add(tf.keras.layers.Conv2D(32,(2,2),activation="relu",
input_shape=(28,28,1)))
model.add(tf.keras.layers.Conv2D(64,(2,2),activation="relu"))
model.add(tf.keras.layers.Conv2D(128,(2,2),2,activation="relu"))
model.add(tf.keras.layers.Conv2D(32,(2,2),activation="relu"))
model.add(tf.keras.layers.Conv2D(64,(2,2),activation="relu"))
model.add(tf.keras.layers.Conv2D(128,(2,2),2,activation="relu"))

# ANN
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile (
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy'],
)

# history 변수에 학습 과정 데이터를 저장
# 로그 확인 가능
history = model.fit(image_train, label_train, 
          validation_data = val_dataset,
          epochs=10, batch_size=10)

# history 변수에 저장된 로그 데이터를 history_mnist_relu 파일으로 저장
# 바이너리 형태의 파일로 저장(wb)
with open('fasion_history_mnist_cnn_relu', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

model.summary()
model.save('fashion_mnist_cnn_relu.h5')