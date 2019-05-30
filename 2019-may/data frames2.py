import pandas as pd
import os

dir = 'E:/'
titanic_train = pd.read_csv(os.path.join(dir, 'train.csv'))

#explore the structure of dataframe
print(titanic_train.shape)
print(titanic_train.columns)
print(titanic_train.dtypes)
print(titanic_train.index)
print(titanic_train.values)
print(titanic_train.info())

#explore sample data
print(titanic_train.head(4))
print(titanic_train.tail(4))
print(titanic_train.sample(n=4))
print(titanic_train.sample(frac=0.1))

#row access with slicing operator or boolean indexing
print(titanic_train[0:3])
print(titanic_train[titanic_train.Sex=='male'])

#column access with single value or list of values
print(titanic_train[ ['Name', 'Age', 'Sex'] ])
print(titanic_train['Name']) #dictionary style access
print(titanic_train.Name) #property style access

#row and column access based on index
titanic_train.iloc[1:3,:]
titanic_train.iloc[1:3,2:4]
titanic_train.iloc[1:3, [True,True]]
#row and column access based on name
titanic_train.loc[1:3, ['Sex','Fare']]
titanic_train.loc[titanic_train.Sex=='male',:]
titanic_train.loc[1:3, :'Fare']

#creating new columns
titanic_train['dummy'] = 1
titanic_train['FamilySize'] = titanic_train['Parch'] + titanic_train['SibSp'] + 1

#explore data like relational sql
#axis=0 means row dimension
#axis=1 means column dimension
#filter rows or columns based on labels
titanic_train.filter(items=['Age', 'Fare'], axis=1).head(3)
titanic_train.filter(items=[5,10,12], axis=0).head()
#filter rows or columns based on conditions
titanic_train.select(lambda x: x.startswith('S'), axis=1).head(3)

#sorting a dataframe
titanic_train.sort_index(ascending=False).head(3)
titanic_train.sort_values(by=['Fare'], ascending=[False]).head()
titanic_train.sort_values(by=['Fare', 'Age'], ascending=[False, True]).head()

#chaining of operations on dataframe
df = (titanic_train.filter(items=['Age', 'Fare']).
      sort_values(by=['Fare'], ascending=[False]).
      head(3))
     
#group by columns
titanic_train.groupby('Sex').size()
titanic_train.groupby(['Sex', 'Pclass']).size()

g = titanic_train.groupby('Sex')
print(g.mean())
print(g['Fare'].mean())
def f(x):
   return x+10
print(g.get_group('male').head(3))

x = pd.core.groupby.DataFrameGroupBy()

