# %%
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
from IPython.core.interactiveshell import InteractiveShell
from keras.api import Model, Sequential, layers
from keras.api.datasets import fashion_mnist

InteractiveShell.ast_node_interactivity = "all"

saved_path: Path = Path.home() / "build" / "model"
saved_path.mkdir(parents=True, exist_ok=True)
# %%
### Title: CNN
(f_image_train, f_label_train), (f_image_test, f_label_test) = fashion_mnist.load_data()
# normalized iamges
f_image_train, f_image_test = f_image_train / 255.0, f_image_test / 255.0

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

plt.figure(figsize=(10, 10))
for i in range(10):
    plt.subplot(3, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(f_image_train[i])
    plt.xlabel(class_names[f_label_train[i]])
    plt.show()

model = Sequential()
model.add(layers.Conv2D(32, (2, 2), activation="relu", input_shape=(28, 28, 1)))
model.add(layers.Conv2D(64, (2, 2), activation="relu"))
model.add(layers.Conv2D(128, (2, 2), 2, activation="relu"))
model.add(layers.Conv2D(32, (2, 2), activation="relu"))
model.add(layers.Conv2D(64, (2, 2), activation="relu"))
model.add(layers.Conv2D(128, (2, 2), 2, activation="relu"))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(f_image_train, f_label_train, epochs=10, batch_size=10)
model.summary()
# %%
model.save(f"{saved_path}/fashion_mnist.keras")  # Correct method to save the model
keras.saving.save_model(model, f"{saved_path}/fashion_mnist.keras")  # Correct usage

# Optionally, using keras.saving.save_model (though the above line already saves it)
loaded_model: Model = keras.saving.load_model(f"{saved_path}/fashion_mnist.keras")

num = 10000
predict = model.predict(f_image_test[:num])
print(f_label_train[:num])
print(" * Prediction, ", np.argmax(predict, axis=1))

loss_, accuracy_ = loaded_model.evaluate(
    f_image_train[:num], f_label_test[:num], verbose=2
)
print(f"ReLU 모델의 테스트 정확도: {accuracy_ * 100:.2f}%")
