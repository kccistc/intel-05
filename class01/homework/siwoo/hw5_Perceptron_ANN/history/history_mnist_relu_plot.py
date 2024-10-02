import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle

              
history_relu = pickle.load(open('./history_mnist_relu', 'rb'))
history_sig = pickle.load(open('./history_mnist_sigmoid', 'rb'))


tra_acc_relu = history_relu["accuracy"]
tra_loss_relu = history_relu['loss']
val_acc_relu = history_relu["val_accuracy"]
val_loss_relu = history_relu['val_loss']

tra_acc_sig = history_sig['accuracy']
val_acc_sig = history_sig['val_accuracy']
tra_loss_sig = history_sig['loss']
val_loss_sig = history_sig['val_loss']


plt.subplot(1,2,1)
plt.title('relu')
plt.plot(range(len(tra_acc_relu)), tra_acc_relu, label = 'tra_acc_relu')
plt.plot(range(len(val_acc_relu)), val_acc_relu, label = 'val_acc_relu')
plt.plot(range(len(tra_acc_sig)), tra_acc_sig, label = 'tra_acc_sig')
plt.plot(range(len(val_acc_sig)), val_acc_sig, label = 'val_acc_sig')
plt.legend()

plt.subplot(1,2,2)
plt.title('sigmoid')
plt.plot(range(len(tra_loss_relu)), tra_loss_relu, label = 'tra_loss_relu')
plt.plot(range(len(val_loss_relu)), val_loss_relu, label = 'val_loss_relu')
plt.plot(range(len(tra_loss_sig)), tra_loss_sig, label = 'tra_loss_sig')
plt.plot(range(len(val_loss_sig)), val_loss_sig, label = 'val_loss_sig')
plt.legend()

plt.show()