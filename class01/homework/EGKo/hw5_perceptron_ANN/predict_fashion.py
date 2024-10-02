#%%
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential, layers, models
from keras.api.datasets import mnist, fashion_mnist

(f_image_t, f_label_t), (f_image_test, f_label_test) = fashion_mnist.load_data()
f_image_t, f_image_test = f_image_t/255.0, f_image_test/255.0

f_image_train = f_image_t[:1000, :, :]  # image_train 이라는 변수는 하나만 만들것. 그래야지 val 카테고리가 만들어짐
f_label_train = f_label_t[:1000]        # label_train도 마찬가지
f_image_val =  f_image_t[1000:1200, :, :]
f_label_val =  f_label_t[1000:1200]
val_dataset = (f_image_val, f_label_val)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

num = 10
model_relu = models.load_model('./mnist_fashion_relu.h5')
model_sig = models.load_model('./mnist_fashion_sigmoid.h5')
predict_relu = model_relu.predict(f_image_test[:num])
predict_sig = model_relu.predict(f_image_test[:num])
print("input image = ", f_label_test[:num])
print("predict relu = ", np.argmax(predict_relu, axis=1))
print("predict sig = ", np.argmax(predict_sig, axis=1))
