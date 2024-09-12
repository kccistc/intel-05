import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle

mnist = tf.keras.datasets.mnist

(image_t, label_t), (image_t, label_t) = mnist.load_data()
image_t, image_t = image_t/255.0, image_t/255.0

image_train = image_t[:1000,:,:]
label_train = label_t[:1000]
image_val = image_t[1000:1200,:,:]
label_val = label_t[1000:1200]
val_dataset = (image_val, label_val)

class_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics = ['accuracy'])

history = model.fit(image_train, label_train, validation_data= val_dataset, epochs = 10, batch_size=10)

with open('history_mnist_relu', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

model.summary()
model.save('mnist_relu.h5')



# %%
mnist = tf.keras.datasets.mnist

(image_t, label_t), (image_test, label_test) = mnist.load_data()
image_t, image_test = image_t/255.0, image_test/255.0

image_train = image_t[:1000,:,:]
label_train = label_t[:1000]
image_val = image_t[1000:1200,:,:]
label_val = label_t[1000:1200]
val_dataset = (image_val, label_val)

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics = ['accuracy'])

history = model.fit(image_train, label_train, validation_data= val_dataset, epochs = 10, batch_size=10)

with open('history_mnist_relu', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

model.summary()
model.save('mnist_relu.h5')

test_loss, test_acc = model.evaluate(image_test, label_test, verbose = 2)
print(f'\n테스트 정확도: {test_acc}')


# %%

# plt.figure(figsize=(10,10))
# for i in range(10):
#     plt.subplot(3,4,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(image_train[i])
#     plt.xlabel(class_names[label_train[i]])
# plt.show()