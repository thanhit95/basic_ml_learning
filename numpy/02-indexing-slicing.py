import numpy as np


a = np.array([[1, 2, 3], [3, 4, 5], [4, 5, 6]])

b = a[1:]

c = a[..., 1]

d = a[..., 1:]

print(c)
print()
print(d)
