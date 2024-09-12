#%%
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential, layers, models
from keras.api.datasets import mnist, fashion_mnist

(image_t, label_t), (image_test, label_test) = mnist.load_data()
image_t, image_test = image_t/255.0, image_test/255.0

image_train = image_t[:1000, :, :]  # image_train 이라는 변수는 하나만 만들것. 그래야지 val 카테고리가 만들어짐
label_train = label_t[:1000]        # label_train도 마찬가지
image_val =  image_t[1000:1200, :, :]
label_val =  label_t[1000:1200]
val_dataset = (image_val, label_val)

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

num = 10
model_relu = models.load_model('./mnist_relu.h5')
model_sig = models.load_model('./mnist_sigmoid.h5')
predict_relu = model_relu.predict(image_test[:num])
predict_sig = model_relu.predict(image_test[:num])
print("input image = ", label_test[:num])
print("predict relu = ", np.argmax(predict_relu, axis=1))
print("predict sig = ", np.argmax(predict_sig, axis=1))
