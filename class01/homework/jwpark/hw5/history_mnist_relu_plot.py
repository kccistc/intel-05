import matplotlib.pyplot as plt
import numpy as np
import pickle

history_relu = pickle.load(open('./history_mnist_relu', 'rb'))
history_sig = pickle.load(open('./history_mnist_sigmoid', 'rb'))

tra_acc_relu = history_relu["accuracy"]
tra_loss_relu = history_relu["loss"]
val_acc_relu = history_relu["val_accuracy"]
val_loss_relu = history_relu["val_loss"]

tra_acc_sig = history_sig["accuracy"]
tra_loss_sig = history_sig["loss"]
val_acc_sig = history_sig["val_accuracy"]
val_loss_sig = history_sig["val_loss"]

# 진행되는 epoch 마다의 accuracy 값을 확인
plt.subplot(1, 2, 1)
plt.title('accuracy')
plt.plot(range(len(tra_acc_relu)), tra_acc_relu, label = 'train (relu)')
plt.plot(range(len(val_acc_relu)), val_acc_relu, label = 'val (relu)')
plt.plot(range(len(tra_acc_sig)), tra_acc_sig, label = 'train (sig)')
plt.plot(range(len(val_acc_sig)), val_acc_sig, label = 'val (sig)')
plt.legend()

# 진행되는 epoch 마다의 loss 값을 확인
plt.subplot(1, 2, 2)
plt.title('loss')
plt.plot(range(len(tra_loss_relu)), tra_loss_relu, label = 'train (relu)')
plt.plot(range(len(val_loss_relu)), val_loss_relu, label = 'val (relu)')
plt.plot(range(len(tra_loss_sig)), tra_loss_sig, label = 'train (sig)')
plt.plot(range(len(val_loss_sig)), val_loss_sig, label = 'val (sig)')
plt.legend()

# 진행되는 epoch 마다의 val_loss 값을 확인
# plt.subplot(1, 3, 3)
# plt.title('val_loss')
# plt.plot(range(len(val_loss)), val_loss, label='val_loss')

plt.show()
plt.savefig('figure.png')