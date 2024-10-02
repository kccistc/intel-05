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

model = keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, (2,2), activation="sigmoid", input_shape=(28,28,1)))
model.add(tf.keras.layers.Conv2D(64, (2,2), activation="sigmoid"))
model.add(tf.keras.layers.Conv2D(128, (2,2), strides=2, activation="sigmoid"))
model.add(tf.keras.layers.Conv2D(32, (2,2), activation="sigmoid"))
model.add(tf.keras.layers.Conv2D(64, (2,2), activation="sigmoid"))
model.add(tf.keras.layers.Conv2D(128, (2,2), strides=2, activation="sigmoid"))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation="sigmoid"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))

model.compile( optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'],)
history = model.fit(image_train, label_train, validation_data = val_dataset, epochs=10)

with open('history_mnist_fashion_nobatch', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

model.summary()
model.save('mnist_fashion_nobatch.h5')

plt.figure(figsize=(10,10))
for i in range(10):
    plt.subplot(3,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(f_image_train[i])
    plt.xlabel(class_names[f_label_train[i]])
plt.show()
