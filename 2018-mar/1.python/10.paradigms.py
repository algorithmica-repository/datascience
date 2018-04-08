from sklearn import tree
import numpy as np
import pandas as pd

#imperative programming in python
a = [10,20]
print(len(a))
print(type(a))
array = np.array([10,20,30])
print(np.max(array))

#oo programming in python
b = [10,20,30]
print(array.shape)
print(type(b))
b.append(40)
dt = tree.DecisionTreeClassifier()
dt.fit()
dt.score()

#functional programming in python
age = [20, 30, 40, 50]
    
for i,e in enumerate(age):
    age[i] = e + 10 

def incr(e):    
    return e+10
age1 = list(map(incr, age))    
#lambda: anonymous function
age2 = list(map(lambda e:e+10, age))
  
#applying map
def dummy(a):
    return a+10
df = pd.DataFrame({'c1':[10,20,30], 'c2':[1,2,3]})
#map method is supported by frame and series types
df['c1'].map(dummy)    
df['c1'].map(lambda a:a+10)

#list doesnot support map 
list1 = [10,20,30]
type(map(dummy, list1))     
       