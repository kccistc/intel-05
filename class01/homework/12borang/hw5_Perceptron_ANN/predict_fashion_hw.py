# %%
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
fashion_mnist = tf.keras.datasets.fashion_mnist
# fashion_mnist = tf.keras.datasets.mnist

(image_t, label_t), (image_test, label_test) = fashion_mnist.load_data()
image_t, image_test = image_t/255.0, image_test/255.0

image_train = image_t[:1000,:,:]
label_train = label_t[:1000]
image_val = image_t[1000:1200,:,:]
label_val = label_t[1000:1200]
val_dataset = (image_val, label_val)

class_names = ['0','1','2','3','4','5','6','7','8','9']

num=10
model_relu = tf.keras.models.load_model('./mnist_fasion_relu.h5')
model_sigmoid = tf.keras.models.load_model('./fashion_mnist_sigmoid.h5')
predict_relu = model_relu.predict(image_test[:num])
predict_sigmoid = model_sigmoid.predict(image_test[:num])

print("input image = ", label_test[:num])
print("predict relu = ", np.argmax(predict_relu, axis=1))
print("predict sigmoid = ", np.argmax(predict_sigmoid, axis=1))


plt.subplot(2,2,1)
plt.plot(predict_relu)

plt.subplot(2,2,2)
plt.plot(predict_sigmoid)

plt.subplot(2,2,3)
plt.plot(label_test)
