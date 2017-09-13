from sklearn import tree
#imperative programming in python
a = [10,20]
print(len(a))
print(type(a))

#oo programming in python
dt = tree.DecisionTreeClassifier()
dt.fit()
dt.score()

#functional programming in python
age = [20, 30, 40, 50]

i = 0
for e in age:
    age[i] = e + 10
    i = i + 1 
    
for i,e in enumerate(age):
    age[i] = e + 10 

def incr(e):
    return e+10
age = list(map(incr, age))    

age = list(map(lambda e:e+10, age))
  
#applying map

def dummy(a):
    return a+10

import pandas as pd

df = pd.DataFrame({'c1':[10,20,30]})
df['c1'].map(dummy)    
df['c1'].map(lambda a:a+10)

list1 = [10,20,30]
type(map(dummy, list1))     
       
       
       
       