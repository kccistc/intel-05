import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.fashion_mnist
#fashion_mnist = tf.keras.datasets.fashion_mnist

(image_t, label_t),(image_test, label_test) = mnist.load_data()
image_t, image_test= image_t/255.0, image_test/255.0

image_train = image_t[:1000,:,:] #1~1000번째 데이터셋을 1000개 사용한다 전체:6만개
label_train =label_t [:1000] 
image_val = image_t[1000:1200,:,:]
lavel_val = label_t [1000:1200]
val_dataset = (image_val,lavel_val)

#class_names = ['0','1','2','3','4','5','6','7','8','9']


num= 10
model_relu = tf.keras.models.load_model('./fashion_relu.h5')
model_sigmoid = tf.keras.models.load_model('./fashion_sigmoid.h5')
predict_relu = model_relu.predict(image_test[:num])
predict_sigmoid = model_sigmoid.predict(image_test[:num])

print("input image = ", label_test[:num])
print("predict relu = ",np.argmax( predict_relu, axis=1))
print("predict sigmoid = ",np.argmax( predict_sigmoid, axis=1))
