import matplotlib.pyplot as plt
import numpy as np
import pickle

relu_history = pickle.load(open('./history_fashion_mnist_relu','rb'))
sigmoid_history = pickle.load(open('./history_fashion_mnist_sigmoid', 'rb'))

relu_acc = relu_history["accuracy"]
relu_loss = relu_history["loss"]

sigmoid_acc = sigmoid_history["accuracy"]
sigmoid_loss = sigmoid_history["loss"]

relu_val_acc = relu_history["val_accuracy"]
relu_val_loss = relu_history["val_loss"]

sigmoid_val_acc = sigmoid_history["val_accuracy"]
sigmoid_val_loss = sigmoid_history["val_loss"]

plt.subplot(1,2,1)
plt.title('accuracy')
plt.plot(range(len(relu_acc)), relu_acc, label = 'relu_accuracy')
plt.plot(range(len(relu_val_acc)), relu_val_acc, label = 'relu_val_accuracy')
plt.plot(range(len(sigmoid_acc)), sigmoid_acc, label = 'sigmoid_accuracy')
plt.plot(range(len(sigmoid_val_acc)), sigmoid_val_acc, label = 'sigmoid_val_accuracy')
plt.legend()
plt.subplot(1,2,2)
plt.title('loss')
plt.plot(range(len(relu_loss)), relu_loss, label = 'relu_loss')
plt.plot(range(len(relu_val_loss)), relu_val_loss, label = 'relu_val_loss')
plt.plot(range(len(sigmoid_loss)), sigmoid_loss, label = 'sigmoid_loss')
plt.plot(range(len(sigmoid_val_loss)), sigmoid_val_loss, label = 'sigmoid_val_loss')
plt.legend()
plt.show()