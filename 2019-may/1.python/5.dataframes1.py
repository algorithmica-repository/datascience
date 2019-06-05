#dataframe: 2-d container with heterogeneous columns
import pandas as pd
dir(pd)

persons = {'name':['abc','def','xyz'],
           'age':[30, 25, 15],
           'fare':[12.5, 40.3, 50]
           }
df1 = pd.DataFrame(persons)
print(df1)
print(df1.shape)
print(df1.info())

name =['abc','def','xyz']
age = [30, 25, 15]
fare = [12.5, 40.3, 50]
df1 = pd.DataFrame({'name': name, 'age':age, 'fare':fare})
print(df1)
print(df1.shape)
print(df1.info())

print(df1.index)
print(df1.values)
df1 = df1.set_index(df1.age)
df1 = df1.reset_index(drop=True)


titanic_train = pd.read_csv('C:/Users/Algorithmica/Downloads/titanic_train.csv')
print(titanic_train.shape)
print(titanic_train.info())
