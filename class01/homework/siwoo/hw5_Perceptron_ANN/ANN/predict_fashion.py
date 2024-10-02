import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import keras
import pickle


fashion_mnist = tf.keras.datasets.fashion_mnist
mnist = tf.keras.datasets.mnist


(f_image_train, f_label_train), (f_image_test, f_label_test) = fashion_mnist.load_data()
f_image_train, f_image_test = f_image_train / 255.0, f_image_test / 255.0

image_train = f_image_train[:1000,:,:]
label_train = f_label_train[:1000]
image_val = f_image_train[1000:1200,:,:]
label_val = f_label_train[1000:1200]
val_dataset = (image_val, label_val)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

num = 10
model_fashion = tf.keras.models.load_model('mnist_fashion.h5')

predict_fashion = model_fashion.predict(f_image_test[:num])

print("f_input_image = ", f_label_test[:num])

print("predict_fashion = ", np.argmax(predict_fashion, axis=1))