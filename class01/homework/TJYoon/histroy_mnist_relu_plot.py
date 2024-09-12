import matplotlib.pyplot as plt
import numpy as np
import pickle 

history_sigmoid = pickle.load(open('./fashion_mnist_sigmoid', 'rb'))
acc_sigmoid = history_sigmoid["accuracy"]
loss_sigmoid = history_sigmoid["loss"]
val_acc_sigmoid = history_sigmoid["val_accuracy"]
val_loss_sigmoid = history_sigmoid["val_loss"]


history_lelu = pickle.load(open('./fashion_mnist_lelu', 'rb'))
acc_lelu = history_lelu["accuracy"]
loss_lelu = history_lelu["loss"]
val_acc_lelu = history_lelu["val_accuracy"]
val_loss_lelu = history_lelu["val_loss"]

plt.subplot(1,2,1)
plt.title('accuracy')
plt.plot(range(len(acc_lelu)), acc_lelu, label = 'relu_accuracy') #accuracy를 그리는데 accuracy 길이에 맞게 
plt.plot(range(len(val_acc_lelu)), val_acc_lelu, label = 'val_relu_accuracy') #accuracy를 그리는데 accuracy 길이에 맞게 
plt.plot(range(len(acc_sigmoid)), acc_sigmoid, label = 'sigmoid_accuracy') #accuracy를 그리는데 accuracy 길이에 맞게 
plt.plot(range(len(val_acc_sigmoid)), val_acc_sigmoid, label = 'val_sigmoid_accuracy') #accuracy를 그리는데 accuracy 길이에 맞게 
plt.legend()

plt.subplot(1,2,2)
plt.title('accuracy')
plt.plot(range(len(loss_sigmoid)), loss_sigmoid, label = 'sigmoid_loss') #accuracy를 그리는데 accuracy 길이에 맞게 
plt.plot(range(len(val_loss_sigmoid)), val_loss_sigmoid, label = 'val_sigmoid_loss') #accuracy를 그리는데 accuracy 길이에 맞게 
plt.plot(range(len(loss_lelu)), loss_lelu, label = 'relu_loss') #accuracy를 그리는데 accuracy 길이에 맞게 
plt.plot(range(len(val_loss_lelu)), val_loss_lelu, label = 'val_relu_loss') #accuracy를 그리는데 accuracy 길이에 맞게 
plt.legend()
plt.show()