# %%

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle

fashion_mnist = tf.keras.datasets.fashion_mnist
(f_image_t, f_label_t), (f_image_test, f_label_test) = fashion_mnist.load_data()
f_image_t, f_image_test = f_image_t/255.0, f_image_test/255.0

f_image_train = f_image_t[:2500,:,:]
f_label_train = f_label_t[:2500]

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize=(10,10))
for i in range(10):
    plt.subplot(3,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(f_image_train[i])
    plt.xlabel(class_names[f_label_train[i]])
plt.show()

# %%

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32,(2,2), activation='relu', input_shape=(28,28,1)))
model.add(tf.keras.layers.Conv2D(64,(2,2), activation='relu'))
model.add(tf.keras.layers.Conv2D(128,(2,2),2, activation='relu'))
model.add(tf.keras.layers.Conv2D(32,(2,2), activation='relu'))
model.add(tf.keras.layers.Conv2D(64,(2,2), activation='relu'))
model.add(tf.keras.layers.Conv2D(128,(2,2),2, activation='relu'))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation='sigmoid'))
model.add(tf.keras.layers.Dense(10,activation='softmax'))

# %%

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics = ['accuracy'])

history = model.fit(f_image_train, f_label_train, validation_data= (f_image_test, f_label_test), epochs = 10, batch_size=10)

model.summary()
model.save('fashion_mnist_relu.h5')
with open('historyBatchRelu', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

# %%
