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
titanic_train.iloc[[True,True], [True, False, True]]
#row and column access based on name
titanic_train.loc[1:3, ['Sex','Fare']]
titanic_train.loc[titanic_train.Sex=='male',:]
titanic_train.loc[1:3, :'Fare']

#creating new columns
titanic_train['dummy'] = 1
titanic_train['FamilySize'] = titanic_train['Parch'] + titanic_train['SibSp'] + 1