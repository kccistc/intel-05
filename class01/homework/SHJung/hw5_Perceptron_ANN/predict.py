import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.fashion_mnist

(image_t, label_t), (image_test, label_test) = mnist.load_data()
image_t, image_test = image_t / 255.0, image_test / 255.0

image_train = image_t[:1000,:,:]
label_train = label_t[:1000]
image_val = image_t[1000:1200,:,:]
label_val = label_t[1000:1200]
val_dataset = (image_val, label_val)
#class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

num = 10
model_relu = tf.keras.models.load_model("./mnist_relu.h5")
model_sigmoid = tf.keras.models.load_model("./mnist_sigmoid.h5")
predict_relu = model_relu.predict(image_test[:num])
predict_sigmoid = model_sigmoid.predict(image_test[:num])

print("input image = ", label_test[:num])
print("predict_relu = ", np.argmax(predict_relu, axis = 1))
print("predict_sigmoid = ", np.argmax(predict_sigmoid, axis = 1))

plt.figure(figsize=(10,10))

plt.title("Relu", fontsize = 20, fontweight = 'bold') 
plt.axis('off')
for i in range(num):
    plt.subplot(4,3,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image_test[i])
    plt.xlabel(f"INPUT : {class_names[label_test[i]]} \nPREDICT : {class_names[np.argmax(predict_relu[i])]}")
plt.tight_layout()

plt.figure(figsize=(10,10))
plt.title("Sigmoid", fontsize = 20, fontweight = 'bold')
plt.axis('off')
for i in range(num):
    plt.subplot(4,3,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image_test[i])
    plt.xlabel(f"INPUT : {class_names[label_test[i]]} \nPREDICT : {class_names[np.argmax(predict_sigmoid[i])]}")
plt.tight_layout()
plt.show()        