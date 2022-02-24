import numpy as np


a = np.array([1,0,2,3,1,4,6])
b = np.array([1,0,8,3,1,4,7])

c = sum(a == b)/a.shape[0]

print(c)