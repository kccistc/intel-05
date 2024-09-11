import numpy as np

n = int(input("정수 n을 입력: "))
array1 = np.arange(n*n)
array1 = np.reshape(array1, (n,n))
for i in range(0, n):
    for j in range(1, n+1):
        array1[i, j-1] = i*n + j
print(array1)