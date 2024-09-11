#%%
a = ('A','B')
b = a + ('C',)
print(b)
# %%
import numpy as np
print("정수 n을 입력하시오. : ")
n = int(input())
a = np.arange(n*n)

for i in range(0,n):
    for j in range(1,n+1):
        a[n*i+j-1] = n*i+j

a = a.reshape(n,n)
print(a)
# %%
import numpy as np
print("정수 n을 입력하시오. : ")
n = int(input())
a = np.arange(n*n)

for i in range(0,n):
    for j in range(1,n+1):
        a[n*i+j-1] = n*i+j

a = a.reshape(n,n)
b = a.reshape(1,n*n)
print(a)
print(b)
