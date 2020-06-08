import numpy as np

a = np.array([9, 4, 7, 3, 1, 5, 8, 0, 100, -2, 5, 8]).reshape(3, 4)

print(a)

print()

print(a[..., 1:3])
