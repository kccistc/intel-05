# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from IPython.core.interactiveshell import InteractiveShell
from keras import Model
from keras.api.layers import (
    Conv2D,
    Dense,
    Flatten,
    Input,
    RandomFlip,
    RandomRotation,
    Rescaling,
)
from keras.api.models import Sequential
from keras.api.utils import image_dataset_from_directory
from PIL import Image  # for reading images

InteractiveShell.ast_node_interactivity = "all"

# Dataset Path
saved_path: Path = Path.home() / "build" / "model"
saved_path.mkdir(parents=True, exist_ok=True)
file_name_stem: str = "fashion_mnist_batch_sig"
dataset_path = Path.home() / "datasets" / "chest_xray"
dataset_path

mainDIR = os.listdir(f"{dataset_path}")
print(mainDIR)
train_folder = f"{dataset_path}/train/"
val_folder = f"{dataset_path}/val/"
test_folder = f"{dataset_path}/test/"

# View a random normal and pneumonia image from the train set
train_n = train_folder + "NORMAL/"
train_p = train_folder + "PNEUMONIA/"

print(len(os.listdir(train_n)))
rand_norm = np.random.randint(0, len(os.listdir(train_n)))
norm_pic = os.listdir(train_n)[rand_norm]
print("normal picture title: ", norm_pic)
norm_pic_address = train_n + norm_pic

rand_p = np.random.randint(0, len(os.listdir(train_p)))
sic_pic = os.listdir(train_p)[rand_p]
sic_address = train_p + sic_pic
print("pneumonia picture title:", sic_pic)

norm_load = Image.open(norm_pic_address)
sic_load = Image.open(sic_address)

f = plt.figure(figsize=(10, 6))
a1 = f.add_subplot(1, 2, 1)
img_plot = plt.imshow(norm_load)
a1.set_title("Normal")

a2 = f.add_subplot(1, 2, 2)
img_plot = plt.imshow(sic_load)
a2.set_title("Pneumonia")
plt.show()


# Load the dataset using image_dataset_from_directory
batch_size = 32
image_size = (64, 64)

train_dataset = image_dataset_from_directory(
    directory=train_folder,
    image_size=image_size,
    batch_size=batch_size,
    label_mode="binary",
)

val_dataset = image_dataset_from_directory(
    val_folder,
    image_size=image_size,
    batch_size=batch_size,
    label_mode="binary",
)

test_dataset = image_dataset_from_directory(
    directory=test_folder,
    image_size=image_size,
    batch_size=batch_size,
    label_mode="binary",
)

# %%
### Title: ANN
# CNN Model Building
model_in = Input(shape=(64, 64, 3))
model = Flatten()(model_in)
model = Dense(activation="relu", units=128)(model)
model = Dense(activation="sigmoid", units=1)(model)

# Compile the Neural Network
model_fin = Model(inputs=model_in, outputs=model)
model_fin.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Data Augmentation (instead of using ImageDataGenerator)
data_augmentation = Sequential(
    [
        Rescaling(1.0 / 255),  # Rescale pixel values
        RandomFlip("horizontal"),  # Randomly flip the images horizontally
        RandomRotation(0.2),  # Randomly rotate the images
    ]
)

# Apply the augmentation to the training dataset
train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y))

# Model Summary
model_fin.summary()

# Model Training
cnn_model = model_fin.fit(
    train_dataset,
    epochs=10,
    validation_data=val_dataset,
)

# Evaluate and Save the Model
test_accu = model_fin.evaluate(test_dataset)
model_fin.save(f"{saved_path}/medical_ann.h5")
print("The testing accuracy is:", test_accu[1] * 100, "%")

# Predictions
Y_pred = model_fin.predict(test_dataset)
y_pred = np.argmax(Y_pred, axis=1)
max(y_pred)

# %%
### Title: CNN

# Functional API로 CNN 모델 구성 (이진 분류)
model_in = Input(shape=(64, 64, 3))

# Conv2D layers
model = Conv2D(32, (3, 3), activation="relu")(model_in)
model = Conv2D(64, kernel_size=(3, 3), activation="relu")(model)
model = Conv2D(128, (3, 3), strides=2, activation="relu")(model)
model = Conv2D(32, (3, 3), activation="relu")(model)
model = Conv2D(64, (3, 3), activation="relu")(model)
model = Conv2D(128, (3, 3), strides=2, activation="relu")(model)

# Flatten & Dense layers
model = Flatten()(model)
model = Dense(128, activation="relu")(model)
model = Dense(1, activation="sigmoid")(model)  # 이진 분류이므로 sigmoid 사용

# Model 정의
model_fin = Model(inputs=model_in, outputs=model)

# 모델 컴파일 (이진 분류에 맞춰 binary_crossentropy 사용)
model_fin.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


# Data Augmentation (instead of using ImageDataGenerator)
data_augmentation = Sequential(
    [
        Rescaling(1.0 / 255),  # Rescale pixel values
        RandomFlip("horizontal"),  # Randomly flip the images horizontally
        RandomRotation(0.2),  # Randomly rotate the images
    ]
)

# Apply the augmentation to the training dataset
train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y))

# Model Summary
model_fin.summary()

# Model Training
cnn_model = model_fin.fit(
    train_dataset,
    epochs=10,
    validation_data=val_dataset,
)

# Evaluate and Save the Model
test_accu = model_fin.evaluate(test_dataset)
model_fin.save(f"{saved_path}/medical_ann.h5")
print("The testing accuracy is:", test_accu[1] * 100, "%")

# Predictions
Y_pred = model_fin.predict(test_dataset)
y_pred = np.argmax(Y_pred, axis=1)
max(y_pred)


# %%
