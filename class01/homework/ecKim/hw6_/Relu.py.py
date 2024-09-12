import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
import numpy as np
import pickle

fashion_mnist = tf.keras.datasets.fashion_mnist

(image_t, label_t), (image_test,label_test)=fashion_mnist.load_data()
image_t, image_test = image_t/255.0,image_test/255.0 #실수로 나오게 하기 위해서
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


image_train=image_t[:1000,:,:]
label_train=label_t[:1000]
image_val=image_t[1000:1200,:,:]
label_val=label_t[1000:1200]
val_dataset = (image_val,label_val)

# plt.figure(figsize=(10,10))
# for i in range(10):
#     plt.subplot(3,4,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(image_train[i])
#     plt.xlabel(class_names[label_train[i]])
# plt.show()

# CNN
model = Sequential()
model.add(tf.keras.layers.Conv2D(32,(2,2),activation="relu",
input_shape=(28,28,1)))
model.add(tf.keras.layers.Conv2D(64,(2,2),activation="relu"))
model.add(tf.keras.layers.Conv2D(128,(2,2),2,activation="relu"))
model.add(tf.keras.layers.Conv2D(32,(2,2),activation="relu"))
model.add(tf.keras.layers.Conv2D(64,(2,2),activation="relu"))
model.add(tf.keras.layers.Conv2D(128,(2,2),2,activation="relu"))
# ANN
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation="relu"))
model.add(tf.keras.layers.Dense(10,activation="softmax"))

model.compile(
optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy'],
)

# history=model.fit(image_train, label_train, epochs=10, batch_size=10)
history=model.fit(image_train, label_train,validation_data=val_dataset, epochs=10, batch_size=10)

with open('history_fashion_mnist_CNN_relu_batch','wb') as file_pi:
    pickle.dump(history.history,file_pi)

model.summary()
model.save('fashion_CNN_relu_batch.h5')
