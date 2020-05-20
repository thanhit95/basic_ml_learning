import numpy as np

a = np.arange(9, dtype=np.int).reshape(3, 3)

print('First array:')
print(a)
print()

print('Second array:')
b = np.array([10,10,10]).reshape(3, 1)
print(b)
print()

print('Add the two arrays:')
print(np.add(a, b))
print()

print('Subtract the two arrays:')
print(np.subtract(a, b))
print()

print('Multiply the two arrays:')
print(np.multiply(a, b))
print()

print('Divide the two arrays:')
print(np.divide(a, b))
