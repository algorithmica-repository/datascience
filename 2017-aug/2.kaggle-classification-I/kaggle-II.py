import pandas as pd
print(pd.__version__)
titanic_train = pd.read_csv("D:/titanic/train.csv")
print(type(titanic_train))

#explore the dataframe
titanic_train.shape
titanic_train.info()

#access columns of a dataframe
titanic_train['Sex']
titanic_train['Fare']
titanic_train.Sex

titanic_train['Survived']
titanic_train.groupby(['Sex', 'Survived']).size()

titanic_test = pd.read_csv('D:/titanic/test.csv')
titanic_test.shape
titanic_test.info()

titanic_test.loc[titanic_test.Sex=='male','Survived'] = 0
titanic_test.loc[titanic_test.Sex=='female','Survived'] = 1
titanic_test.to_csv('D:/titanic/submission.csv', columns=['PassengerId','Survived'],index=False)


