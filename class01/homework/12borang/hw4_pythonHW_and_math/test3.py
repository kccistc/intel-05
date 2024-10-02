# %%
import numpy as np
a = int(input())
x = np.arrange(a*a) + 1
print(x)
b = np.reshape(x,(a,a))
print(b)