import numpy as np
import matplotlib.pylab as plt

def f(x):
    return x**2 - 4*x + 6

Number0fpoints = 101
x = np.linspace(-5, 5, Number0fpoints)
fx = f(x)

plt.plot(x,fx)
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('plot of f(x)')
plt.show()

# %%

a = 1
b = 2
x = (a, b)
c = 3
x = x + (c,)

print(f'티플 값 = {x}')

# %%

n = int(input())
a = 0
x = ()

for i in range(1, n):
    for j in range(n, n+1):
        print('{i}')
        print('{j}')