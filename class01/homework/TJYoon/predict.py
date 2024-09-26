import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.fashion_mnist
#fashion_mnist = tf.keras.datasets.fashion_mnist

(image_train_, label_train_),(image_test, label_test) = mnist.load_data()
image_train_, image_test = image_train_/255.0, image_test/255.0

image_train = image_train_[:1000, : , :]
label_train = label_train_[:1000]
image_validation = image_train_[1000 : 1200, : , :]
label_validation = label_train_[1000: 1200]

val_dataset = (image_validation, label_validation)

class_names = ['0','1','2','3','4','5','6','7','8','9']

num = 10
model_relu = tf.keras.models.load_model('fashion_mnist_lelu.h5')
model_sigmoid = tf.keras.models.load_model('fashion_mnist_sigmoid.h5')

predict_relu = model_relu.predict(image_test[:num])
predict_sigmoid = model_relu.predict(image_test[:num])

print("input image = ", label_test[:num])
print("predict relu = ", np.argmax(predict_relu,axis = 1))
print("predict sigm = ", np.argmax(predict_sigmoid,axis = 1))
