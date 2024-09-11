import numpy as mp

N = input()

arr = [[0]*int(N)] * int(N)

num = 0

for i in range(0, int(N)):
    for j in range(0, int(N)):
        num += 1
        arr[i][j] = num

data = mp.array(arr).reshape(int(N)*int(N))

print(data)