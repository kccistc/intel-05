import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
import matplotlib.pyplot as plt
import numpy as np
import pickle

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

# CNN
model = Sequential()
model.add(Conv2D(32,(2,2),activation="sigmoid",
input_shape=(28,28,1)))
model.add(Conv2D(64,(2,2),activation="sigmoid"))
model.add(Conv2D(128,(2,2),2,activation="sigmoid"))
model.add(Conv2D(32,(2,2),activation="sigmoid"))
model.add(Conv2D(64,(2,2),activation="sigmoid"))
model.add(Conv2D(128,(2,2),2,activation="sigmoid"))

# ANN
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='sigmoid'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics= ['accuracy'] )
history = model.fit(image_train, label_train, validation_data = val_dataset, epochs=10, batch_size=10)

with open('history_cnn_no_batch', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

model.summary()
model.save('cnn_no_batch.h5')    
