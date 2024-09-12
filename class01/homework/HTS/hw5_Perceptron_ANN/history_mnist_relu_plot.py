import matplotlib.pyplot as plt
import numpy as np
import pickle 

history_relu = pickle.load(open('./history_fashion_mnist_relu', 'rb'))
history_sigmoid = pickle.load(open('./history_fashion_mnist_sigmoid', 'rb'))

tra_acc_relu = history_relu["accuracy"]
tra_loss_relu = history_relu["loss"]
val_acc_relu = history_relu["val_accuracy"]
val_loss_relu = history_relu["val_loss"]

tra_acc_sigmoid = history_sigmoid["accuracy"]
tra_loss_sigmoid = history_sigmoid["loss"]
val_acc_sigmoid = history_sigmoid["val_accuracy"]
val_loss_sigmoid = history_sigmoid["val_loss"]

plt.subplot(1,2,1)
plt.title('accuracy')
plt.plot(range(len(tra_acc_relu)), tra_acc_relu, label = 'train (relu)')
plt.plot(range(len(val_acc_relu)), val_acc_relu, label = 'val (relu)')
plt.plot(range(len(tra_acc_sigmoid)),tra_acc_sigmoid , label = 'train (sig)')
plt.plot(range(len(val_acc_sigmoid)), val_acc_sigmoid, label = 'val (sig)')
plt.legend()

plt.subplot(1,2,2)
plt.title('loss')
plt.plot(range(len(tra_loss_relu)), tra_loss_relu, label = 'train (relu)')
plt.plot(range(len(val_loss_relu)), val_loss_relu, label = 'val (relu)')
plt.plot(range(len(tra_loss_sigmoid)),tra_loss_sigmoid , label = 'train (sig)')
plt.plot(range(len(val_loss_sigmoid)), val_loss_sigmoid, label = 'val (sig)')
plt.legend()
plt.show()