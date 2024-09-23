import pickle 
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

history_relu = pickle.load(open('./history_fashion_relu','rb'))
history_sig = pickle.load(open('./history_fashion_sigmoid','rb'))

tra_acc_relu = history_relu["accuracy"]
tra_loss_relu = history_relu["loss"]
val_acc_relu = history_relu["val_accuracy"]
val_loss_relu = history_relu["val_loss"]

tra_acc_sig = history_sig["accuracy"]
tra_loss_sig = history_sig["loss"]
val_acc_sig = history_sig["val_accuracy"]
val_loss_sig = history_sig["val_loss"]

plt.subplot(1,2,1)
plt.title('Accuracy')
plt.plot(range(len(tra_acc_relu)),tra_acc_relu,label='tra_A_relu')
plt.plot(range(len(val_acc_relu)),val_acc_relu,label='val_A_relu')
plt.plot(range(len(tra_acc_sig)),tra_acc_sig,label='tra_A_sig')
plt.plot(range(len(val_acc_sig)),val_acc_sig,label='val_A_sig')
plt.legend()

plt.subplot(1,2,2)
plt.title('Loss')
plt.plot(range(len(tra_loss_relu)),tra_loss_relu,label='tra_L_relu')
plt.plot(range(len(val_loss_relu)),val_loss_relu,label='val_L_relu')
plt.plot(range(len(tra_loss_sig)),tra_loss_sig,label='tra_L_sig')
plt.plot(range(len(val_loss_sig)),val_loss_sig,label='val_L_sig')
plt.legend()

plt.show()