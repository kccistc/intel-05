#%%
#과제1
AB_tuple = ('A', 'B')
C_value = 'C'
ABC_tuple = AB_tuple + (C_value,)
print(ABC_tuple)

#%%
#과제2
n = int(input("숫자 입력:"))
matrix = []
num = 1
for i in range(n):
    row = []
    for j in range(n):
        row.append(num)
        num += 1
    matrix.append(row)
for row in matrix:
    for num in row:
        print(num, end=" ")
    print()

# for i in range(n):
#     for j in range(n):
#         print(n*i+j + 1,end=" ")
#     print("")
#%%
#과제3
import numpy as np

n = int(input('숫자입력:'))

matrix = []
num = 1
for i in range(n):
    row = []
    for j in range(n):
        row.append(num)
        num += 1
    matrix.append(row)

np_matrix = np.array(matrix)

one_dimensional_array = np_matrix.reshape(-1)

print(one_dimensional_array)