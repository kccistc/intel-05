# 실습2
# 정수 n을 입력받아 n x n 크기의 숫자 사각형을 출력

import numpy as np

num = int(input("1 이상의 정수 n을 입력하세요:"))
ary = np.arange(1, num*num+1)

n = 0
for i in range(num):
    for j in range(num):
        print(ary[n], end=' ')
        n+=1
    print()