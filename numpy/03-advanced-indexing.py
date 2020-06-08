import numpy as np

# x = np.array([[1, 2], [3, 4], [5, 6]])

# y = x[[0,1,2], [0,1,0]]

# print(y)


x = np.array([[ 0,  1,  2], [ 3,  4,  5], [ 6,  7,  8], [ 9, 10, 11]]) 

print('Our array is:')
print(x)
print()

rows = np.array([[0,0], [2,3]])
cols = np.array([[0,2], [0,0]]) 
y = x[rows,cols]

print('The corner elements of this array are:')
print(y)