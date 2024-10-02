import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle
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

model = Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation = 'sigmoid'))
model.add(tf.keras.layers.Dense(64, activation = 'sigmoid'))
model.add(tf.keras.layers.Dense(10, activation = 'softmax'))

model.compile( optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(image_tranin, label_train, validation_data=val_data, epochs=10, batch_size=10)

with open('history_fashion_mnist_sigmoid', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

model.summary()
model.save('fashion_mnist_sigmoid.h5')

#plt.figure(figsize=(10,10))
#for i in range(10):S
#    plt.xticks([])
#    plt.yticks([])
#    plt.grid(False)
#    plt.imshow(image_tranin[i])
#    plt.xlabel(class_names[label_train[i]])
#plt.show()