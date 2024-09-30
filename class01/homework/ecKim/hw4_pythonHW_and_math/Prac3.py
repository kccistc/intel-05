import numpy as np
import matplotlib.pylab as plt


n=10
arr=np.zeros((n,n))
for i in range (n):
    for j in range (n):
        arr[i][j]=i*j

arr2=arr.reshape((n*n,1))




print(arr2)