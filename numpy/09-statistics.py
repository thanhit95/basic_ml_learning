import numpy as np

# a = np.array([9, 4, 7, 3, 1, 6, 8, 0, 100, -2, 5, 8])
a = np.array([727.7, 1086.5, 1091.0, 1361.3, 1490.5, 1956.1])

print('minimum of array:', np.amin(a))
print('maximum of array:', np.amax(a))
print()

b = np.sort(a)
print(b)
print()


print('mean:', np.mean(b))
print('median:', np.median(b))  # (5 + 6) / 2

print('population standard deviation:', np.std(b))          # độ lệch chuẩn
print('population standard variation:', np.var(b))          # phương sai

print('sample standard deviation:', np.std(b, ddof=1))      # độ lệch chuẩn có điều chỉnh
print('sample standard variation:', np.var(b, ddof=1))      # phương sai có điều chỉnh

print()

