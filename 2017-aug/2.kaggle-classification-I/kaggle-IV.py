import pandas as pd

titanic_train = pd.read_csv("D:/titanic/train.csv")
print(type(titanic_train))

#explore the dataframe
titanic_train.shape
titanic_train.info()

titanic_train['Survived']
titanic_train.groupby('Pclass').size()
#find pattern for which females are 0? and which males are 1?
titanic_train.groupby(['Sex','Pclass','Survived']).size()
titanic_train.groupby(['Sex','Embarked','Survived']).size()
titanic_train.groupby(['Sex','Pclass','Embarked','Survived']).size()

titanic_test = pd.read_csv('D:/titanic/test.csv')
titanic_test.shape
titanic_test.info()
titanic_test['Survived'] = 0

titanic_test.loc[titanic_test.Pclass==1,'Survived'] = 1
#titanic_test.loc[titanic_test.Pclass==2,'Survived'] = 0
#titanic_test.loc[titanic_test.Pclass==3,'Survived'] = 0
titanic_test.to_csv('D:/titanic/submission.csv', columns=['PassengerId','Survived'],index=False)


