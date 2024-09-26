


n = int(input('정수 n을 입력하면 사각형 출력: '))

nxn = [[i * n + j + 1 for j in range(n)] for i in range(n)]

for row in nxn:
    print(row)