import pandas as pd

titanic_train = pd.read_csv('C:/Users/Algorithmica/Downloads/titanic_train.csv')
print(type(titanic_train))
titanic_train.shape
titanic_train.info()

#access columns
titanic_train['Pclass']
titanic_train.Pclass
titanic_train[['Pclass','Survived']]
#better alternative for multi column access
columns = ['Pclass', 'Survived']
titanic_train[columns]

#access rows
titanic_train[0:3]
titanic_train[:10]
titanic_train[800:]

#access row and columns
titanic_train.loc[0:3,columns]
titanic_train.loc[:3,columns]
titanic_train.iloc[0:3,1:3]

#condional access on columns
titanic_train.iloc[titanic_train.Sex == 'female', 3]
titanic_train.loc[titanic_train.Sex == 'female', 'Sex']
