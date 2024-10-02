import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle 
#Practice 1 - make model

mnist = tf.keras.datasets.fashion_mnist
#fashion_mnist = tf.keras.datasets.fashion_mnist

(image_train_, label_train_),(image_test, label_test) = mnist.load_data()
image_train_, image_test = image_train_/255.0, image_test/255.0

#validation data 만들기
## label은 1차원!

image_train = image_train_[:1000, : , :]
label_train = label_train_[:1000]
image_validation = image_train_[1000 : 1200, : , :]
label_validation = label_train_[1000: 1200]

val_dataset = (image_validation, label_validation)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

tf.keras.Sequential(layers=None, trainable=True, name=None)

plt.figure(figsize=(10,10))
for i in range(10):
    plt.subplot(3,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image_train[i])
    plt.xlabel(class_names[label_train[i]])

plt.show()
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='sigmoid'))
model.add(tf.keras.layers.Dense(64, activation='sigmoid'))
model.add(tf.keras.layers.Dense(10,activation='softmax'))

model.compile(
optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy'],
)

# history = log 뽑아냄.

history = model.fit(image_train, label_train, 
          validation_data = val_dataset,
          epochs=10, batch_size=10)

with open('fashion_mnist_sigmoid', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

model.summary()
model.save('fashion_mnist_sigmoid.h5')

#Practice 1 - Load model

# model = tf.keras.models.load_model('./fashion_mnist.h5')
# fashion_mnist = tf.keras.datasets.fashion_mnist
# (f_image_train, f_label_train), (f_image_test, f_label_test) = fashion_mnist.load_data()
# f_image_train, f_image_test = f_image_train / 255.0, f_image_test / 255.0
# num = 10
# predict = model.predict(f_image_test[:num])
# print(f_label_train[:num])
# print(" * Prediction, ", np.argmax(predict, axis = 1))