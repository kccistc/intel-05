import matplotlib.pyplot as plt
import pickle

# class01_ANN_01.py 에서 pickle로 저장한 history를 로드할 예정
history_relu = pickle.load(open('./history_fashion_mnist_relu', 'rb'))
history_sig = pickle.load(open('./history_fashion_mnist_sigmoid', 'rb'))

tra_acc_relu = history_relu["accuracy"]
tra_loss_relu = history_relu["loss"]
val_acc_relu = history_relu["val_accuracy"]
val_loss_relu = history_relu["val_loss"]

tra_acc_sig = history_sig["accuracy"]
tra_loss_sig = history_sig["loss"]
val_acc_sig = history_sig["val_accuracy"]
val_loss_sig = history_sig["val_loss"]


# epoch가 진행되면서 accuracy와 loss가 어떻게 변화하는지 볼 것
plt.subplot(1, 2, 1)
plt.title('accuracy')
plt.plot(range(len(tra_acc_relu)), tra_acc_relu, label = 'train (relu)')
plt.plot(range(len(val_acc_relu)), val_acc_relu, label = 'val (relu)')
plt.plot(range(len(tra_acc_sig)), tra_acc_sig, label = 'train (sig)')
plt.plot(range(len(val_acc_sig)), val_acc_sig, label = 'val (sig)')
plt.legend()
plt.subplot(1, 2, 2)
plt.title('loss')
plt.plot(range(len(tra_loss_relu)), tra_loss_relu, label = 'train (relu)')
plt.plot(range(len(val_loss_relu)), val_loss_relu, label = 'val (relu)')
plt.plot(range(len(tra_loss_sig)), tra_loss_sig, label = 'train (sig)')
plt.plot(range(len(val_loss_sig)), val_loss_sig, label = 'val (sig)')
plt.legend()
plt.show()