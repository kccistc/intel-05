import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
import numpy as np

#mnist = tf.keras.datasets.mnist

# (image_t, label_t), (image_test, label_test)= mnist.load_data()
# image_t, image_test = image_t/255.0, image_test/255.0

# # 6만개의 데이터 셋 중 1000개만 추출
# # image_train 데이터는 번호, width, height로 구성됨
# image_train = image_t[:1000,:,:]
# label_train = label_t[:1000]
# image_val = image_t[1000:1200,:,:]
# label_val = label_t[1000:1200]
# val_dataset = (image_val, label_val)

# class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# # 이미지 번호
# num = 10
# model_relu = tf.keras.models.load_model('./mnist_relu.h5')
# model_sigmoid = tf.keras.models.load_model('./mnist_sigmoid.h5')
# predit_relu = model_relu.predict(image_test[:num])
# predit_sigmoid = model_sigmoid.predict(image_test[:num])

# # 예측 픽셀 값 출력
# # np.argmax: 해당 픽셀 값에 대해 가장 일치 확률이 높은 확률을 가지는 픽셀 값을 도출하기 위함
# print("predict relu = ", np.argmax(predit_relu, axis=1))
# print("predict sigmoid = ", np.argmax(predit_sigmoid, axis=1))


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

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 이미지 번호
num = 10
model_relu = tf.keras.models.load_model('./fashion_mnist_relu.h5')
model_sigmoid = tf.keras.models.load_model('./fashion_mnist_sigmoid.h5')
predit_relu = model_relu.predict(image_test[:num])
predit_sigmoid = model_sigmoid.predict(image_test[:num])

# 예측 픽셀 값 출력
# np.argmax: 해당 픽셀 값에 대해 가장 일치 확률이 높은 확률을 가지는 픽셀 값을 도출하기 위함
print("predict relu = ", np.argmax(predit_relu, axis=1))
print("predict sigmoid = ", np.argmax(predit_sigmoid, axis=1))