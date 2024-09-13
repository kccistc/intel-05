import numpy as np  # for linear algebra
import matplotlib.pyplot as plt  # for plotting things
import os
from PIL import Image  # for reading images
# Keras Libraries <- CNN
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Input, BatchNormalization, Concatenate, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

# Set up directories
mainDIR = os.listdir('./chest_xray')
print(mainDIR)
train_folder = './chest_xray/train/'
val_folder = './chest_xray/val/'
test_folder = './chest_xray/test/'

# Training folders
train_n = train_folder + 'NORMAL/'
train_p = train_folder + 'PNEUMONIA/'

# Display a random normal and pneumonia image
print(len(os.listdir(train_n)))
rand_norm = np.random.randint(0, len(os.listdir(train_n)))
norm_pic = os.listdir(train_n)[rand_norm]
print('normal picture title: ', norm_pic)
norm_pic_address = train_n + norm_pic

rand_p = np.random.randint(0, len(os.listdir(train_p)))
sic_pic = os.listdir(train_p)[rand_p]
sic_address = train_p + sic_pic
print('pneumonia picture title:', sic_pic)

# Load the images
norm_load = Image.open(norm_pic_address)
sic_load = Image.open(sic_address)

# Plot images
f = plt.figure(figsize=(10, 6))
a1 = f.add_subplot(1, 2, 1)
plt.imshow(norm_load)
a1.set_title('Normal')
a2 = f.add_subplot(1, 2, 2)
plt.imshow(sic_load)
a2.set_title('Pneumonia')
plt.show()

# Data augmentation and normalization
num_of_test_samples = 600
batch_size = 32
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('./chest_xray/train',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

validation_generator = test_datagen.flow_from_directory('./chest_xray/val/',
                                                        target_size=(64, 64),
                                                        batch_size=32,
                                                        class_mode='binary')

test_set = test_datagen.flow_from_directory('./chest_xray/test',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

# Define the CNN model
model_fin = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # For binary classification
])

# Compile the model
model_fin.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

# Show model summary
model_fin.summary()

