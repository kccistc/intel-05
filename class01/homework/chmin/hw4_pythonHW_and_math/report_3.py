# import numpy as np

# insert_num = int(input("숫자 입력하세요: "))
# num = 1
# for row in range(insert_num):
#     for col in range(insert_num):
#         print(num, end="\t")
#         num += 1
#     print("")
# list_num = list(range(1,num))
# print(list_num)
#과제3
import numpy as np

n = int(input("정수를 입력하세요: "))
matrix = np.arange(1, n*n + 1).reshape(n, n)
print('\n'.join(' '.join(map(str, row)) for row in matrix))
matrix_1 = np.arange(1, n*n + 1).reshape(1,n*n)
print('\n'.join(' '.join(map(str, row)) for row in matrix_1))