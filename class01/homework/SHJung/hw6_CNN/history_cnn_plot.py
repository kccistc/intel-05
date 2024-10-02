import matplotlib.pyplot as plt
import numpy as np
import pickle

history_batchSig = pickle.load(open('./historyBatchSig', 'rb'))
history_noBatchSig = pickle.load(open('./historyNoBatchSig', 'rb'))
history_batchReLu = pickle.load(open('./historyBatchReLu', 'rb'))

val_acc_relu = history_batchReLu["val_accuracy"]
val_loss_relu = history_batchReLu["val_loss"]

val_acc_nbsig = history_noBatchSig["val_accuracy"]
val_loss_nbsig = history_noBatchSig["val_loss"]

val_acc_bsig = history_batchSig["val_accuracy"]
val_loss_bsig = history_batchSig["val_loss"]

plt.subplot(1, 2, 1)
plt.title('Validation Loss')
plt.plot(range(len(val_loss_nbsig)), val_loss_nbsig, label = 'sigmoid (No Batch)')
plt.plot(range(len(val_loss_bsig)), val_loss_bsig, label = 'sigmoid (Batch)')
plt.plot(range(len(val_loss_relu)), val_loss_relu, label = 'relu')
plt.grid()
plt.legend()

plt.subplot(1, 2, 2)
plt.title('Validation Accuracy')
plt.plot(range(len(val_acc_nbsig)), val_acc_nbsig, label = 'sigmoid (No Batch)')
plt.plot(range(len(val_acc_bsig)), val_acc_bsig, label = 'sigmoid (Batch)')
plt.plot(range(len(val_acc_relu)), val_acc_relu, label = 'relu')
plt.grid()
plt.legend()
plt.savefig('results_cnn.png')
plt.show()




