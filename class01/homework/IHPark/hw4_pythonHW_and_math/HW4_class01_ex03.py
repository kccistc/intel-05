# 실습3
# 실습1에서 수행한 결과를 reshape을 이용해서 1차원 형태로 변환한다.

import numpy as np

num = int(input("1 이상의 정수 n을 입력하세요:"))
ary = np.arange(1, num*num+1)

print(ary,'\n')

n = 0
for i in range(num):
    for j in range(num):
        print(ary[n], end=' ')
        n+=1
    print()
print()

ary.reshape(1, num*num)
print(ary)