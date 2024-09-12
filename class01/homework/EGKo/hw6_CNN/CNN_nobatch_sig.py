#%%
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential, layers
from keras.api.datasets import mnist, fashion_mnist
import pickle

##expandedDim, repeat

(f_image_t, f_label_t), (f_image_test, f_lebel_test) = fashion_mnist.load_data()
f_image_t, f_image_test = f_image_t/255.0, f_image_test/255.0

f_image_train = f_image_t[:1000, :, :]  # image_train 이라는 변수는 하나만 만들것. 그래야지 val 카테고리가 만들어짐
f_label_train = f_label_t[:1000]        # label_train도 마찬가지
f_image_val =  f_image_t[1000:1200, :, :]
f_label_val =  f_label_t[1000:1200]
val_dataset = (f_image_val, f_label_val)

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

# CNN
model = Sequential()
model.add(layers.Conv2D(32,(2,2),activation="sigmoid",
input_shape=(28,28,1)))
model.add(layers.Conv2D(64,(2,2),activation="sigmoid"))
model.add(layers.Conv2D(128,(2,2),2,activation="sigmoid"))
model.add(layers.Conv2D(32,(2,2),activation="sigmoid"))
model.add(layers.Conv2D(64,(2,2),activation="sigmoid"))
model.add(layers.Conv2D(128,(2,2),2,activation="sigmoid"))

# ANN
model.add(layers.Flatten())
model.add(layers.Dense(128,activation="sigmoid"))
model.add(layers.Dense(10,activation="softmax"))

model.compile(optimizer  = 'adam', 
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

history = model.fit(f_image_train, f_label_train,
                    validation_data = val_dataset,
                    epochs = 10, batch_size = 10)

# with open('history_fashion_mnist_relu', 'wb') as file_pi: # wb = write binary
with open('history_CNN_nobatch_sigmoid', 'wb') as file_pi: # wb = write binary
    pickle.dump(history.history, file_pi)

model.summary()
# model.save('mnist_fashion_relu.h5')
model.save('CNN_nobatch_sigmoid.h5')