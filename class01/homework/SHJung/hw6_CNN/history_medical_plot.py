import matplotlib.pyplot as plt
import numpy as np
import pickle

history_medical = pickle.load(open('./historyMedicalCnn', 'rb'))

acc_relu = history_medical["accuracy"]
loss_relu = history_medical["loss"]
val_acc_relu = history_medical["val_accuracy"]
val_loss_relu = history_medical["val_loss"]

plt.subplot(1, 2, 1)
plt.title('Loss')
plt.plot(range(len(loss_relu)), loss_relu, label = 'loss_relu')
plt.plot(range(len(val_loss_relu)), val_loss_relu, label = 'val_loss_relu')
plt.grid()
plt.legend()

plt.subplot(1, 2, 2)
plt.title('Accuracy')
plt.plot(range(len(acc_relu)), acc_relu, label = 'acc_relu')
plt.plot(range(len(val_acc_relu)), val_acc_relu, label = 'val_acc_relu')
plt.grid()
plt.legend()
plt.savefig('results_medical.png')
plt.show()




