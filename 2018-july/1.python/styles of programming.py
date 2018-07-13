from sklearn import tree
import numpy as np
import pandas as pd

a = 10
print(type(a))
#imperative programming in python
a = [10,20]
print(len(a))
print(type(a))
array = np.array([10,20,30])
print(np.max(array))

#oo programming in python
b = [10,20,30]
print(type(b))
b.append(40)
dt = tree.DecisionTreeClassifier()
dt.fit()
dt.score()

#functional programming in python
age = [20, 30, 40, 50]
age = np.array(age) + 10

for e in age:
     print(e+10)    
     
for i,e in enumerate(age):
    age[i] = e + 10 

def incr(e):    
    return e+10
age1 = list(map(incr, age))    
#lambda: anonymous function
age2 = list(map(lambda e:e+10, age))
  