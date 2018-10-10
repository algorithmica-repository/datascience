import pandas as pd

print(pd.__version__)

#creation of data frames from csv
titanic_train = pd.read_csv("C:\\Users\\Algorithmica\\Downloads\\titanic_train.csv")
print(type(titanic_train))
print(titanic_train.info())

#access columns
titanic_train['Age']
features = ['Age', 'Pclass']
titanic_train[features]

#access rows
titanic_train[0:3]
titanic_train[800:]

#access rows & columns(index & name based)
titanic_train.iloc[0:3,0:2]
titanic_train.iloc[10:13,0:]
titanic_train.loc[0:3,features]

#conditional access
titanic_train[titanic_train['Pclass']==1]
titanic_train.loc[titanic_train['Pclass']==1, features]
