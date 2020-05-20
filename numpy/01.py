import numpy as np

# a = np.array([[1, 2, 3], [4, 5, 6]])
a = np.arange(24)

print(a.ndim)
print(np.shape(a))

b = np.reshape(a, (3, 8))
print(b)

print()

c = np.zeros((2, 3), dtype=int)
print(c)
