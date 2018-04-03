import numpy as np

list1 = [10, 20, 30]
list2 = [1, 2, 3]

print(list1+list2)
print(list1**2)

#creation of arrays
a1 = np.array(list1)
a2 = np.array(list2)
print(a1 + a2)
print(a1 * a2)
print(type(a1))
print(a1.shape)

#access elements
a1[0]
a1[0:2]
a1[0:]

#relational operations on array
a1 > 20
#conditional access of elements
a1[a1>20]

#common operations on array
a1.min()
a1.mean()
a1.argmin()
a1.sort()

list1 = [10, True, 'abc']
array1 = np.array(list1)

list2 = [10, True]
array2 = np.array(list2)
