import pickle 
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.fashion_mnist
#fashion_mnist = tf.keras.datasets.fashion_mnist

(image_t, label_t),(image_test, label_test) = mnist.load_data()
image_t, image_test= image_t/255.0, image_test/255.0

image_train = image_t[:1000,:,:] #1~1000번째 데이터셋을 1000개 사용한다 전체:6만개
label_train =label_t [:1000] 
image_val = image_t[1000:1200,:,:]
lavel_val = label_t [1000:1200]
val_dataset = (image_val,lavel_val)

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

model = tf.keras.Sequential()

# CNN
model.add(tf.keras.layers.Conv2D(32,(2,2),activation="relu",
input_shape=(28,28,1)))
model.add(tf.keras.layers.Conv2D(64,(2,2),activation="relu"))
model.add(tf.keras.layers.Conv2D(128,(2,2),2,activation="relu"))
model.add(tf.keras.layers.Conv2D(32,(2,2),activation="relu"))
model.add(tf.keras.layers.Conv2D(64,(2,2),activation="relu"))
model.add(tf.keras.layers.Conv2D(128,(2,2),2,activation="relu"))

# ANN
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation="sigmoid"))
model.add(tf.keras.layers.Dense(10,activation="softmax"))

plt.show()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
    
)

history = model.fit(image_train, label_train, validation_data = val_dataset, epochs=10, batch_size=10)
with open('history_fashion_Relu_s','wb') as file_pi:
    pickle.dump(history.history, file_pi)

model.summary()
model.save('fashion_Relu_s.h5')