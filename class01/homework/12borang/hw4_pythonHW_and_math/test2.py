# %%
n = int(input())
j = 0
i = 0
num = 1

for i in range(n):
    for j in range(n):
        print(num, end=" ")
        num = num + 1
    print("")