import numpy as np
import matplotlib.pylab as plt


n=10
arr=np.zeros((n,n))
for i in range (n):
    for j in range (n):
        arr[i][j]=i*j

# arr=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])

print(arr)