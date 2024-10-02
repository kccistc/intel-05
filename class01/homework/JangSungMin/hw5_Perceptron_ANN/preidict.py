import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import Sequential

mnist = tf.keras.datasets.mnist
#fashion_mnist = tf.keras.datasets.fashion_mnist
(image_t, label_t), (image_test, label_test) = mnist.load_data()

image_tranin = image_t[:1000,:,:]
label_train = label_t[:1000]
image_val = image_t[1000:1200,:,:]
label_val = label_t[1000:1200]
val_data = (image_val, label_val)
image_tranin, image_test = image_tranin/255.0, image_test/255.0

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankleboot']

num = 10
model_relu = tf.keras.models.load_model('./fashion_mnist_relu.h5')
model_sigmoid = tf.keras.models.load_model('./fashion_mnist_sigmoid.h5')
predict_relu = model_relu.predict(image_test[:num])
predict_sigmoid = model_sigmoid.predict(image_test[:num])
print("input image = ", label_test[:num])
print("predict relu = ", np.argmax(predict_relu, axis=1))
print("predict sigmoid = ", np.argmax(predict_sigmoid, axis=1))