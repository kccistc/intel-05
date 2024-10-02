import numpy as np

n = int(input())
count = 1
row = []

for i in range(n):
    for j in range(n):
       row.append(count)
       count = count + 1

row = np.array(row).reshape(1, n*n)
print(row)
