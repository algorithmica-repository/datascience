import numpy as np

print(np.__version__)

#creation of an array from homogeneous list
a1 = np.array([[10,20], [30,40]])
print(type(a1))
print(a1.shape)

a2 = np.array([[1,2], [3,4]])
print(a2.shape)

#2-d array operations
print(np.max(a1,1))
print(np.argmax(a1,0))
print(np.mean(a1, 0))
print(a1.cumsum())

print(a1+a2)
print(a1*a2)
print(a1-a2)
print(a1.dot(a2))
print(a1.T)



