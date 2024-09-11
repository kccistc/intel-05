# *초기값 0, learning rate을 1.2로 한 경우

import numpy as np
import matplotlib.pylab as plt

# 함수 정의
def f(x):
    return x**2 - 4*x + 6

def grad_fx(x):
    return 2*x - 4

# 경사 하강법 함수
def steepest_descent(func, grad_func, x0, learning_rate=0.01, MaxIter=10, verbose=True):
    paths = []
    for i in range(MaxIter):
        x1 = x0 - learning_rate * grad_func(x0)  # 학습률과 기울기 곱해야 함
        if verbose:
            print('{0:03d} : {1:4.3f}, {2:4.2E}'.format(i, x1, func(x1)))  # format 함수 수정
        x0 = x1
        paths.append(x0)
    return x0, func(x0), paths

# 초기값 설정 및 경사 하강법 실행
xopt, fopt, paths = steepest_descent(f, grad_fx, 1.2, learning_rate=1.2, MaxIter=10, verbose=True)

# 그래프 그리기
x = np.linspace(0.5, 2.5, 1000)
paths = np.array(paths)
plt.plot(x, f(x))  # 함수 그래프
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('plot of f(x)')

plt.plot(paths, f(paths), 'o-')  # 경로 그래프
plt.show()

plt.plot(f(paths), 'o-')
plt.grid()
plt.xlabel('x')
plt.ylabel('cost')
plt.title('plot of cost')
plt.show()