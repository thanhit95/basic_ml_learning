import numpy as np

a = np.arange(1, 13)
b = a.T

print(a.shape)
print(a)
print()
print(b.shape)
print(b)


a = np.arange(1, 13).reshape(3, 4)
b = a.T

print(a.shape)
print(a)
print()
print(b.shape)
print(b)
