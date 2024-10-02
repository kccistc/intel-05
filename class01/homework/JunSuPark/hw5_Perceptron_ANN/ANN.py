# %%

import matplotlib.pyplot as plt
import numpy as np


# 시그모이드 함수 정의
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def numerical_derivative(f, x):
    delta_x = 1e-4  ## ? 1/1000
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index  #
        tmp_val = x[idx]

        x[idx] = float(tmp_val) + delta_x
        fx1 = f(x)  # f(x + delta_x)

        x[idx] = float(tmp_val) - delta_x
        fx2 = f(x)  # f(x - delta_x)

        grad[idx] = (fx1 - fx2) / (2 * delta_x)

        x[idx] = tmp_val
        it.iternext()

    return grad


# 논리 게이트 클래스 정의
class logicGate:
    def __init__(self, gate_name, xdata, tdata, learning_rate=0.01, threshold=0.5):
        self.name = gate_name
        self.__xdata = xdata.reshape(4, 2)
        self.__tdata = tdata.reshape(4, 1)

        self.__w = np.random.rand(2, 1)
        self.__b = np.random.rand(1)

        self.__learning_rate = learning_rate
        self.__threshold = threshold

    # 손실 함수 정의
    def __loss_func(self):
        delta = 1e-7
        z = np.dot(self.__xdata, self.__w) + self.__b
        y = sigmoid(z)
        return -np.sum(
            self.__tdata * np.log(y + delta)
            + (1 - self.__tdata) * np.log((1 - y) + delta)
        )

    # 에러 계산 함수 정의
    def err_val(self):
        delta = 1e-7
        z = np.dot(self.__xdata, self.__w) + self.__b
        y = sigmoid(z)
        return -np.sum(
            self.__tdata * np.log(y + delta)
            + (1 - self.__tdata) * np.log((1 - y) + delta)
        )

    # 학습 함수 정의
    def train(self):
        f = lambda x: self.__loss_func()
        print("init error:", self.err_val())

        for stp in range(20000):
            self.__w -= self.__learning_rate * numerical_derivative(f, self.__w)
            self.__b -= self.__learning_rate * numerical_derivative(f, self.__b)

            if stp % 2000 == 0:
                print("step:", stp, "error:", self.err_val())

        return f

    # 예측 함수 정의
    def predict(self, input_data):
        z = np.dot(input_data, self.__w) + self.__b
        y = sigmoid(z)

        if y[0] > self.__threshold:
            result = 1
        else:
            result = 0

        return y, result


# 학습 데이터 정의 (AND 게이트)
xdata = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
tdata = np.array([[0, 0, 0, 1]]).T  # 정답 데이터

# AND 게이트 생성 및 학습
AND = logicGate("AND", xdata, tdata)
AND.train()

# AND 게이트 예측
for in_data in xdata:
    (sig_val, logic_val) = AND.predict(in_data)
    print(in_data, ":", logic_val)

# 학습 데이터 정의 (OR 게이트)
xdata = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
tdata = np.array([[0, 1, 1, 1]]).T  # 정답 데이터

# OR 게이트 생성 및 학습
OR = logicGate("OR", xdata, tdata)
OR.train()

# OR 게이트 예측
for in_data in xdata:
    (sig_val, logic_val) = OR.predict(in_data)
    print(in_data, ":", logic_val)


# %%
import numpy as np

# Define x and y values using linspace
x = np.linspace(-2, 2, 11)
y = np.linspace(-2, 2, 11)

# Print the x and y values
print(x)
print(y)

# Create a meshgrid for x and y
x, y = np.meshgrid(x, y)
print(x)
print(y)

# Define the function f
f = lambda x, y: (x - 1) ** 2 + (y - 1) ** 2
z = f(x, y)
print(z)

# Compute the gradient of f
grad_f_x = lambda x, y: 2 * (x - 1)
grad_f_y = lambda x, y: 2 * (y - 1)

dz_dx = grad_f_x(x, y)
dz_dy = grad_f_y(x, y)

# Plot the contour and gradient field
ax = plt.axes()
ax.contour(x, y, z, levels=np.linspace(0, 10, 20), cmap=plt.cm.jet)
ax.quiver(x, y, -dz_dx, -dz_dy)

# Set axis to equal and add grid
ax.axis("equal")
ax.grid()

# Label axes
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")

# Show plot
plt.show()


"tf.keras.utils.image_dataset_from_directory"
