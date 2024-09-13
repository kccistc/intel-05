# %%
import pickle
from pathlib import Path

import keras
import matplotlib.pyplot as plt
from keras.api import Sequential, layers
from keras.api.datasets import mnist

saved_path: Path = Path.home() / "build" / "model"
saved_path.mkdir(parents=True, exist_ok=True)

(image_t, label_t), (image_test, label_test) = mnist.load_data()
# 이미지 데이터를 255로 나누어 각 픽셀의 값을 0과 1 사이로 정규화합니다. 이는 신경망 모델의 학습 성능을 향상시키는 일반적인 전처리 작업입니다.
image_t, image_test = image_t / 255.0, image_test / 255.0

image_train = image_t[:1000, :, :]
label_train = label_t[:1000]
image_val = image_t[1000:1200, :, :]
label_val = label_t[1000:1200]
val_dataset = (image_val, label_val)


# class_names = [str(object=i) for i in range(10)]
class_names = list(range(10))

plt.figure(figsize=(10, 10))
for i in range(10):
    plt.subplot(3, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image_train[i])
    plt.xlabel(class_names[label_train[i]])
plt.show()


activations = ["relu", "sigmoid"]
for activation in activations:
    model = Sequential(
        layers=[
            layers.Flatten(),
            layers.Dense(128, activation=f"{activation}", name="layer2"),
            layers.Dense(64, activation=f"{activation}", name="layer3"),
            ### homework
            layers.Dense(10, activation="softmax", name="layer4"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        image_train, label_train, validation_data=val_dataset, epochs=10, batch_size=10
    )

    with open(f"history_mnist_{activation}", "wb") as file_pi:
        pickle.dump(history.history, file_pi)

    # Print model summary
    model.summary()

    # Save the model in Keras format
    model.save(f"mnist_{activation}.keras")  # Correct method to save the model

    # Optionally, using keras.saving.save_model (though the above line already saves it)
    keras.saving.save_model(model, f"mnist_{activation}.keras")  # Correct usage

history_relu = pickle.load(open("./history_mnist_relu", "rb"))
history_sig = pickle.load(open("./history_mnist_sigmoid", "rb"))

tra_acc_relu = history_relu["accuracy"]
tra_loss_relu = history_relu["loss"]
val_acc_relu = history_relu["val_accuracy"]
val_loss_relu = history_relu["val_loss"]

tra_acc_sig = history_sig["accuracy"]
tra_loss_sig = history_sig["loss"]
val_acc_sig = history_sig["val_accuracy"]
val_loss_sig = history_sig["val_loss"]


plt.subplot(1, 2, 1)
plt.title("accuracy")
plt.plot(range(len(tra_acc_relu)), tra_acc_relu, label="train (relu)")
plt.plot(range(len(val_acc_relu)), val_acc_relu, label="val (relu)")
plt.plot(range(len(tra_acc_sig)), tra_acc_sig, label="train (sig)")
plt.plot(range(len(val_acc_sig)), val_acc_sig, label="val (sig)")
plt.legend()
plt.subplot(1, 2, 2)
plt.title("loss")
plt.plot(range(len(tra_loss_relu)), tra_loss_relu, label="train (relu)")
plt.plot(range(len(val_loss_relu)), val_loss_relu, label="val (relu)")
plt.plot(range(len(tra_loss_sig)), tra_loss_sig, label="train (sig)")
plt.plot(range(len(val_loss_sig)), val_loss_sig, label="val (sig)")
plt.legend()

plt.show()
