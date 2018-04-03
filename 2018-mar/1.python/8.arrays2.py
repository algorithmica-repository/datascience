import numpy as np

#create 2-d array
a1 = np.array([[1,2,3], [4,5,6]])
type(a1)
a1.shape

#access elements
a1[1,1]
#access elements by slicing
a1[1,:]
a1[:,1]
a1[0:1,1]

#Element wise operations on matrices
a2 = np.array([[1,2],[3,4]])
a3 = np.array([[1,1],[2,2]])
print(a2 + a3)
print(a2 * a3)

#matrix multiplication
a2.dot(a3)
#matrix transpose
a2.T
#reshape the matrix
a5 = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])
a5.shape
#reshape return modified array. It does not modify array in-place
a5.reshape((2, 3))
tmp1 = a5.reshape((1,6))
tmp1.shape

#Reshape a matrix into a single row, figuring out the correct number of columns
a6 = a5.reshape((1, -1))
a6.shape
type(a6)

#getting useful statistics on matrix
a1.max(axis=0)
a1.max(axis=1)
a1.mean(axis=1)
