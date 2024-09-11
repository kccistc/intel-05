#과제2
# insert_num = int(input("숫자 입력하세요: "))

# list_num = []
# num = 1
# for row in range(insert_num):
#     for col in range(insert_num):
#         list_num.append(num)
#         num = num + 1
#         list_num.append('\t')
#     list_num.append('\n')
# for i in range(len(list_num)):
#     print(list_num[i],end="")


import numpy as np

n = int(input("정수를 입력하세요: "))
matrix = np.arange(1, n*n + 1).reshape(n, n)
print('\n'.join(' '.join(map(str, row)) for row in matrix))