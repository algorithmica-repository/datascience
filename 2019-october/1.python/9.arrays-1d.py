#why list cant be used for array?
list1 = [10, 20, 30]
list2 = [1, 2, 3]
print(list1 + list2)
print(list1*10)

#array: an indexed group of homogeneous elements
import numpy as np

dir(np)

#creating array from list
x1 = np.array([1,2,3,4,5])
print(x1)
print(type(x1))
print(x1.shape)
print(x1.dtype)

x2 = np.array([1,2,3], dtype=np.int64)
print(x2)
print(type(x2))
print(x2.shape)
print(x2.dtype)

#creating array from ranges
x3 = np.linspace(1, 10, 10)
print(x3)
x4 = np.logspace(0, 1, 20)
print(x4)
x5 =  np.arange(0, 20)
print(x5)

res = range(1,10)
print(type(res))
print(res)
for i in res:
    print(i)
tmp = list(res)
print(list(range(1,10,3)))


#accessing elements of 1-d array
print(x1[0])
#using slicing
print(x1[0:3])
#using slicing with stride
print(x1[::2])

#operations on 1-d arrays
a1 = np.array([10,20,30,40])
a2 = np.array([1,2,3,4])
print(a1+a2)
print(a1*a2)
print(a1-a2)
print(a1*3)

print(np.max(a1))
print(np.argmax(a1))
print(np.mean(a1))
print(a1.mean())
print(np.cumsum(a1))
print(np.dot(a1, a2))
print(a1.reshape((2,2)))
a2 = a1.reshape((2,2))

np.log(a2)
