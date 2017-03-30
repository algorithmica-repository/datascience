def add1(a,b):
    return a+b

def add2(a,b,c):
    tmp = a+b
    return tmp+c

def add3(a,b,c=20,d=40):
    tmp = a+b+c
    return tmp+d

print(add1(10,20))
print(add2(10,20,30))
print(add3(3,4,d=50))
print(add3(3,4,10))

def dummy(a):
    return a+10

import pandas as pd
df = pd.DataFrame({'c1':[10,20,30]})
df['c1'].map(dummy)    
df['c1'].map(lambda a:a+10)

list1 = [10,20,30]
type(map(dummy, list1))
