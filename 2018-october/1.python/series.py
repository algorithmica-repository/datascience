#series a 1-d group of elements of same type
import pandas as pd

#create custom frames with dictionary

df1 = pd.DataFrame({'age':[10,20,30],
                    'pclass':[1,2,3]})
print(type(df1))
print(df1)

#create custom series
s1 = pd.Series([10,20,30,40])
print(type(s1))
print(s1)

#get series data from dataframe
tmp1 = df1['age']
print(type(tmp1))

#get subframe out of dataframe
features = ['age']
tmp2 = df1[features]
type(tmp2)
     
#operations on series
print(tmp1[0])
print(tmp1[0:3])
tmp1[0] = 50

#iterate through seris
for x in tmp1:
    print(x)
    
for i,x in enumerate(tmp1):
    print(i, x)

