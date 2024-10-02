import numpy as np
import matplotlib.pylab as plt

# 데이터 생성
np.random.seed(320)
x_train = np.linspace(-1, 1, 51)
f = lambda x: 0.5 * x + 1.0
y_train = f(x_train) + 0.4 * np.random.rand(len(x_train))

# 데이터 셔플링
np.random.seed(303)
shuffled_id = np.arange(0, len(x_train))
np.random.shuffle(shuffled_id)
x_train = x_train[shuffled_id]
y_train = y_train[shuffled_id]

# 손실 함수
def loss(w, x_set, y_set):
    N = len(x_set)
    val = 0.0
    for i in range(N):
        val += 0.5 * (w[0] * x_set[i] + w[1] - y_set[i])**2
    return val / N

# 손실 함수의 기울기
def loss_grad(w, x_set, y_set):
    N = len(x_set)
    val = np.zeros(len(w))
    for i in range(N):
        er = w[0] * x_set[i] + w[1] - y_set[i]
        val += er * np.array([x_set[i], 1.0])
    return val / N

# 배치 생성
def generate_batches(batch_size, features, labels):
    assert len(features) == len(labels)
    out_batches = []
    sample_size = len(features)
    for start_i in range(0, sample_size, batch_size):
        end_i = start_i + batch_size
        batch = (features[start_i:end_i], labels[start_i:end_i])
        out_batches.append(batch)
    return out_batches

# 하이퍼파라미터 설정
batch_size = 10
lr = 0.01
MaxEpochs = 51
alpha = 0.9

# 경사 하강법 (SGD)
w0 = np.array([4.0, -1.0])
path_sgd = []
for epoch in range(MaxEpochs):
    if epoch % 10 == 0:
        print(epoch, w0, loss(w0, x_train, y_train))
    for x_batch, y_batch in generate_batches(batch_size, x_train, y_train):
        path_sgd.append(w0)
        grad = loss_grad(w0, x_batch, y_batch)
        w0 -= lr * grad

# 모멘텀을 사용한 경사 하강법
w0 = np.array([4.0, -1.0])
path_mm = []
velocity = np.zeros_like(w0)
for epoch in range(MaxEpochs):
    if epoch % 10 == 0:
        print(epoch, w0, loss(w0, x_train, y_train))
    for x_batch, y_batch in generate_batches(batch_size, x_train, y_train):
        path_mm.append(w0)
        grad = loss_grad(w0, x_batch, y_batch)
        velocity = alpha * velocity - lr * grad
        w0 += velocity

# 손실 함수 시각화
W0 = np.linspace(-2, 5, 101)
W1 = np.linspace(-2, 5, 101)
W0, W1 = np.meshgrid(W0, W1)
LOSSW = np.zeros_like(W0)
for i in range(W0.shape[0]):
    for j in range(W0.shape[1]):
        wij = np.array([W0[i, j], W1[i, j]])
        LOSSW[i, j] = loss(wij, x_train, y_train)

# 경로 시각화
fig, ax = plt.subplots(figsize=(6, 6))
ax.contour(W0, W1, LOSSW, cmap=plt.cm.jet, levels=np.linspace(0, max(LOSSW.flatten()), 20))

# SGD 경로
paths = np.array(np.matrix(path_sgd).T)
ax.quiver(paths[0, :-1], paths[1, :-1], paths[0, 1:] - paths[0, :-1], paths[1, 1:] - paths[1, :-1],
          scale_units='xy', angles='xy', scale=1, color='k')

# 모멘텀 경로
paths = np.array(np.matrix(path_mm).T)
ax.quiver(paths[0, :-1], paths[1, :-1], paths[0, 1:] - paths[0, :-1], paths[1, 1:] - paths[1, :-1],
          scale_units='xy', angles='xy', scale=1, color='r')

plt.legend(['GD', 'Momentum'])
plt.show()
