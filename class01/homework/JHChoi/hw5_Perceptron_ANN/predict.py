import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.fashion_mnist


#label까지 받아와야 함
(image_t, label_t), (image_test, label_test) = tf.keras.datasets.fashion_mnist.load_data() 
image_t, image_test = image_t/255.0, image_test/255.0

image_train = image_t[:1000,:,:]
label_train = label_t[:1000]
image_val = image_t[1000:1200,:,:]
label_val = label_t[1000:1200]
val_dataset = (image_val, label_val)

#class_names = ['0','1','2','3','4','5','6','7','8','9']
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

num = 10
model_relu = tf.keras.models.load_model('./mnist_relu.h5')
model_sigmoid = tf.keras.models.load_model('./mnist_sigmoid.h5')
predict_relu = model_relu.predict(image_test[:num])
predict_sigmoid = model_sigmoid.predict(image_test[:num])
print("input image = ", label_test[:num])
print("predict relu = ", np.argmax(predict_relu, axis=1))
print("predict sigmoid = ", np.argmax(predict_sigmoid, axis=1))