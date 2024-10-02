import numpy as np

def print_square(n):
    num = 1
    arr = np.zeros((n, n), dtype=int) 
    for i in range(n):
        for j in range(n):
            arr[i, j] = num  
            num += 1
    return arr

n = int(input("정수를 입력하세요: "))

arr = print_square(n)

print(arr)
