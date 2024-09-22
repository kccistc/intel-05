# %%
import numpy as np
import matplotlib.pylab as plt

def f(x):
    return x**2 - 4*x + 6
# f = lambda x: x**2 - 4*x + 6

NumberOfPoints = 101
x = np.linspace(-5, 5, NumberOfPoints)
fx = f(x)
# print (x)
# print (fx)
plt.plot(x,fx)
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('plot of f(x)')
plt.show()
# %%
import numpy as np
import matplotlib.pylab as plt

def f(x):
    return x**2 - 4*x + 6
# f = lambda x: x**2 - 4*x + 6

NumberOfPoints = 101
x = np.linspace(-5, 5, NumberOfPoints)
fx = f(x)

xid = np.argmin(fx)
xopt = x[xid]
print(xopt, f(xopt))
plt.plot(x,fx)
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('plot of f(x)')

plt.plot(xopt, f(xopt), 'xr')
plt.show()

# %%
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

# %%
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
xopt, fopt, paths = steepest_descent(f, grad_fx, 1.0, learning_rate=1, MaxIter=10, verbose=True)

# 그래프 그리기
x = np.linspace(0.5, 3.5, 1000)
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
# %%
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
xopt, fopt, paths = steepest_descent(f, grad_fx, 1.0, learning_rate=0.001, MaxIter=10, verbose=True)

# 그래프 그리기
x = np.linspace(0.5, 3.5, 1000)
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
# %%
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
xopt, fopt, paths = steepest_descent(f, grad_fx, 3, learning_rate=0.9, MaxIter=10, verbose=True)

# 그래프 그리기
x = np.linspace(0.5, 3.5, 1000)
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
# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from visualize import contour_with_quiver, contour_with_path, surf  # 모듈의 함수 불러오기

# 범위 설정
xmin, xmax, xstep = -4.0, 4.0, 0.25
ymin, ymax, ystep = -4.0, 4.0, 0.25

# 그리드 생성
x, y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep), np.arange(ymin, ymax + ystep, ystep))

# 함수 정의
f = lambda x, y: (x - 2)**2 + (y - 2)**2

# z 값 계산
z = f(x, y)
minima = np.array([2., 2.])  # 최소값

# 최소값에서 함수 값 계산
f(*minima)  # f(minima[0], minima[1])와 동일

# reshape 없이 minima를 전달
print(minima)
surf(f, x, y, minima=minima.reshape(1, -1))  # 필요한 형태로 최소값 전달

# 기울기 정의
grad_f_x = lambda x, y: 2 * (x - 2)
grad_f_y = lambda x, y: 2 * (y - 2)

# 등고선 그래프와 벡터 필드 그리기
contour_with_quiver(f, x, y, grad_f_x, grad_f_y, minima=minima.reshape(1, -1))

# %%
a = 1
b = 2
x = (a,b)
c = 3
x = x+(c,)

print(f'tuple x값 = {x}') 
# %%
n = int(input())
i=0
j=0
num=1
for i in range(n):
    for j in range(n):
        num=num+1
        x = x+[num,]
# %%
import numpy as np
n = int(input())
x = np.arange(n*n) + 1
print(x)
b = np.reshape(x, (n,n))
print(b)
# %%
import numpy as np
n = int(input())
array = np.arange(n*n)
array = np.reshape(array,(n,n))

for i in range(1,n):
    for j in range(1,n):
        array[i,j] = i*n + j+1
        
print(array)
# %%
import numpy as np
import cv2  # OpenCV 라이브러리

image_path = 'lena.jpg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_with_batch = np.expand_dims(image, axis=0)
print("After expand_dims:", image_with_batch.shape)
image_transposed = np.transpose(image_with_batch, (0, 3, 1, 2))
print("After transpose:", image_transposed.shape)
