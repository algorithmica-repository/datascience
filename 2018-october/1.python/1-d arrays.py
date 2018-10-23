import numpy as np

#creation of an array from homogeneous list
a1 = np.array([10,20,30])
print(type(a1))
print(a1.shape)

#array from heterogenous list
a2 = np.array([10,20,30.4])
print(a2)

a3= np.array([10,20,'abc'])
print(a3)

#list vs array
l1 = [10,20,30]
l2 = [1,2,3]
print(l1+l2)
print(l1*l2)
print(l1-l2)
print(l1*3)

a1 =  np.array([10,20,30,40])
a2 = np.array([1,2,3, 4])
print(a1+a2)
print(a1*a2)
print(a1-a2)
print(a1*3)

#1-d array operations
print(a1.argmax())
print(a1.std())
print(a1.cumsum())
print(a1.dot(a2))
print(a1.mean())
print(a1.reshape((2,2)))


