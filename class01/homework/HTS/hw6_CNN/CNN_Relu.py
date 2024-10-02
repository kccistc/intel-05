import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential, layers
from keras.api.datasets import mnist, fashion_mnist
import pickle

(image_t, label_t), (image_test, lebel_test) = fashion_mnist.load_data()
image_t, image_t = image_t/255.0, image_t/255.0

image_train = image_t[:1000, :, :]
label_train = label_t[:1000]
image_val =  image_t[1001:1200, :, :]
label_val =  label_t[1001:1200]
val_dataset = (image_val, label_val)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
plt.figure(figsize=(10,10))
for i in range(10):
    plt.subplot(3,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image_train[i])
    plt.xlabel(class_names[label_train[i]])
plt.show()

model = Sequential()
#CNN
model.add(tf.keras.layers.Conv2D(32,(2,2),activation="relu",input_shape=(28,28,1)))
model.add(tf.keras.layers.Conv2D(64,(2,2),activation="relu"))
model.add(tf.keras.layers.Conv2D(128,(2,2),2,activation="relu"))
model.add(tf.keras.layers.Conv2D(32,(2,2),activation="relu"))
model.add(tf.keras.layers.Conv2D(64,(2,2),activation="relu"))
model.add(tf.keras.layers.Conv2D(128,(2,2),2,activation="relu"))
#ANN
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='sigmoid'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer  = 'adam', 
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

history = model.fit(image_train, label_train,
                    validation_data = val_dataset,
                    epochs = 10, batch_size = 10)

with open('history_fashion_mnist_sigmoid', 'wb') as file_pi: # wb = write binary
    pickle.dump(history.history, file_pi)

model.summary()
model.save('CNN_Relu.h5')  