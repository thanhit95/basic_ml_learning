import numpy as np

a = np.arange(0, 60, 5).reshape(3, 4)

a = a + 1

print('Original array is')
print(a)
print()

print('Each values:')
for x in np.nditer(a[..., 1:]):
    print(x, end=' ')
