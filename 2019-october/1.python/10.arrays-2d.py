import numpy as np

#creating array from lists
x1 = np.array([
                [1,2,3],
                [4,5,6]
            ], dtype=np.int64)
print(x1)
print(type(x1))
print(x1.shape)
print(x1.dtype)

#creating array from ranges
x2 = np.arange(0, 20).reshape((4,5))
print(x2)
print(x2.shape)

#creating special arrays
x3 = np.zeros((3,4))
print(x3)
x4 = np.ones((3,4))
print(x4)
x5 = np.eye(4)
print(x5)

#extracting a scalar
print(x2[1,1])
#extracting a vector
print(x2[1,:])
print(x2[:,1])
print(x2[1:3,1:3])
#using slices with stride
print(x2[::2,::2])

#2-d array operations
print(x3+x4)
print(x3*x4)
print(x3-x4)
print(x3.dot(x4))
print(x2.T)

print(np.max(x2,1))
print(np.max(x2,0))
print(np.argmax(x2,0))
print(np.mean(x2, 0))
print(np.cumsum(x2,0))
