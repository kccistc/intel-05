#%%
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential, layers
from keras.api.datasets import mnist, fashion_mnist
import pickle

(image_t, label_t), (image_test, lebel_test) = mnist.load_data()
image_t, image_test = image_t/255.0, image_test/255.0

image_train = image_t[:1000, :, :]  # image_train 이라는 변수는 하나만 만들것. 그래야지 val 카테고리가 만들어짐
label_train = label_t[:1000]        # label_train도 마찬가지
image_val =  image_t[1000:1200, :, :]
label_val =  label_t[1000:1200]
val_dataset = (image_val, label_val)

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
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
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='sigmoid'))
model.add(layers.Dense(64, activation='sigmoid'))
# model.add(layers.Dense(128, activation='relu'))
# model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer  = 'adam', 
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

history = model.fit(image_train, label_train,
                    validation_data = val_dataset,
                    epochs = 10, batch_size = 10)

# with open('history_mnist_relu', 'wb') as file_pi: # wb = write binary
with open('history_mnist_sigmoid', 'wb') as file_pi: # wb = write binary
    pickle.dump(history.history, file_pi)

model.summary()
# model.save('mnist_sigmoid.h5')
model.save('mnist_sigmoid.h5')