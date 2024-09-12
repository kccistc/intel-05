import tensorflow as tf, keras
import matplotlib.pyplot as plt
import numpy as np


fashion_mnist = tf.keras.datasets.fashion_mnist

(image_t, label_t), (image_test, label_test) = fashion_mnist.load_data()
image_t, image_test = image_t / 255.0, image_test / 255.0


image_train = image_t[:1000,:,:]
label_train = label_t[:1000]
image_val = image_t[1000:1200,:,:]
label_val = label_t[1000:1200]
val_dataset = (image_val, label_val)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankel boot'] # 2. fashion_mnist

num = 10
model_relu = keras.models.load_model('./fashion_mnist_relu.h5')
model_sigmoid = keras.models.load_model('./fashion_mnist_relu.h5')
predict_relu = model_relu.predict(image_test[:num])
predict_sigmoid = model_sigmoid.predict(image_test[:num])
print("input image = \t\t", label_test[:num])
print("predict relu = \t\t", np.argmax(predict_relu, axis=1))
print("predict sigmoid = \t", np.argmax(predict_sigmoid, axis=1))