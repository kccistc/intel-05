import tensorflow as tf, keras
import matplotlib.pyplot as plt
import pickle


fashion_mnist = tf.keras.datasets.fashion_mnist

(image_t, label_t), (image_test, label_test) = fashion_mnist.load_data()
image_t, image_test = image_t / 255.0, image_test / 255.0

image_train = image_t[:1000,:,:]
label_train = label_t[:1000]
image_val = image_t[1000:1200,:,:]
label_val = label_t[1000:1200]
val_dataset = (image_val, label_val)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankel boot']

plt.figure(figsize=(10,10))
for i in range(10):
    plt.subplot(3,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image_train[i])
    plt.xlabel(class_names[label_train[i]])
plt.show()

# CNN
model = keras.Sequential()
model.add(keras.layers.Conv2D(32, (2,2), activation='sigmoid',input_shape=(28,28,1)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(64,(2,2),activation='sigmoid'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(128,(2,2),2,activation='sigmoid'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(32,(2,2),activation='sigmoid'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(64,(2,2),activation='sigmoid'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(128,(2,2),2,activation='sigmoid'))
model.add(keras.layers.BatchNormalization())
# ANN
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='sigmoid'))
model.add(keras.layers.Dense(10, activation='softmax'))


model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)
history = model.fit(image_train, label_train, epochs=10, batch_size=10, validation_data=val_dataset)

with open('history_CNN_batch_fashion_mnist_sigmoid', 'wb') as file_pi: # wb = write binary
    pickle.dump(history.history, file_pi)

model.summary()
model.save('CNN_batch_fashion_mnist_sigmoid.h5')