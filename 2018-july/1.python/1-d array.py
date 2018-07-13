##array is an indexed container of homogeneous elements

import numpy as np
print(np.__version__)

#create arrays
arr1 = np.array([10, 20, 30])
arr2 = np.array([30, 40, 50])
print(arr1 + arr2)
print(arr1 > 30)

#retrieve elements of an array
print(arr1[1])
print(arr1[0:2])
print(arr1.argmax())
print(arr1.mean())
print(arr1.std())
print(arr1.var())
print(arr1.shape)
